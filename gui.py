import FreeSimpleGUI as sg
import torch, librosa, threading, pickle
import numpy as np
from torch.nn import functional as F
from torchaudio.transforms import Resample
from modules.vocoder import load_model, load_onnx_model
from modules.extractors import F0Extractor, VolumeExtractor, UnitsEncoder
from modules.extractors.common import upsample
import sys
import argparse
import time
import gui_locale
if len(sys.argv) <= 1:
    import sounddevice as sd

flag_vc = False

def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = a * (fade_out ** 2) + b * (fade_in ** 2) + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    return result


class SvcDDSP:
    def __init__(self) -> None:
        self.model = None
        self.units_encoder = None
        self.encoder_type = None
        self.encoder_ckpt = None
        self.enhancer = None
        self.enhancer_type = None
        self.enhancer_ckpt = None
        self.spk_info = None
        self.spk_embeds = None
        self.pitch_extractor = None
        self.select_pitch_extractor = None
        self.pitch_extractor_sample_rate = None
        self.args = None

    def update_model(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load ddsp model
        if self.model is None or self.model_path != model_path:
            model_path_ext = model_path.split('.')[-1]
            if model_path_ext == 'onnx':
                self.model, self.args, self.spk_info = load_onnx_model(
                        model_path, providers=['CPUExecutionProvider'])  # TODO: make providers selectable
                self.device = 'cpu'
            else:
                self.model, self.args, self.spk_info = load_model(model_path, device=self.device)
            self.model_path = model_path

            # load units encoder
            if self.units_encoder is None or self.args.data.encoder != self.encoder_type or self.args.data.encoder_ckpt != self.encoder_ckpt:
                self.units_encoder = UnitsEncoder(
                    self.args.data.encoder,
                    self.args.data.encoder_ckpt,
                    self.args.data.encoder_sample_rate,
                    self.args.data.encoder_hop_size,
                    device=self.device,
                    extract_layers=self.args.model.units_layers)
                self.encoder_type = self.args.data.encoder
                self.encoder_ckpt = self.args.data.encoder_ckpt
                
            if self.spk_info is not None:
                # update speaker embeds
                self.spk_embeds = {
                    i: {
                        'spk_embed': torch.from_numpy(self.spk_info[i].item()['spk_embed']).float().to(self.device).unsqueeze(0),
                        'spk_name': self.spk_info[i].item()['name'],
                    }
                    for i in self.spk_info.files
                }
            else:
                self.spk_embeds = None
                
        return self.device
    
    
    def update_pitch_extractor(self, pitch_extractor_type, sample_rate):
        if self.args is not None:
            hop_size = self.args.data.block_size * sample_rate / self.args.data.sampling_rate
            self.pitch_extractor = F0Extractor(
                pitch_extractor_type,
                sample_rate,
                hop_size,
                self.args.data.f0_min,
                self.args.data.f0_max)
        self.select_pitch_extractor = pitch_extractor_type
        self.pitch_extractor_sample_rate = sample_rate
                

    def infer(self,
              audio,
              sample_rate,
              spk_id=1,
              threhold=-45,
              pitch_adjust=0,
              use_spk_mix=False,
              spk_mix_dict=None,
              pitch_extractor_type='crepe',
              f0_min=50,
              f0_max=1100,
              intonation=1.0,
              intonation_base=220.0,
              safe_prefix_pad_length=0,
              ):
        # print("Infering...")
        # load input
        # audio, sample_rate = librosa.load(input_wav, sr=None, mono=True)
        hop_size = self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        # safe front silence
        if safe_prefix_pad_length > 0.03:
            silence_front = safe_prefix_pad_length - 0.03
        else:
            silence_front = 0

        # extract f0
        # pitch_extractor = F0Extractor(
        #     pitch_extractor_type,
        #     sample_rate,
        #     hop_size,
        #     self.args.data.f0_min,
        #     self.args.data.f0_max)
        # f0 = self.pitch_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front)
        f0 = self.pitch_extractor.extract(audio, uv_interp=False, device=self.device, silence_front=silence_front)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0_uv = f0 == 0
        f0[f0_uv] = torch.rand_like(f0[f0_uv])*float(self.args.data.sampling_rate/self.args.data.block_size) + float(self.args.data.sampling_rate/self.args.data.block_size)
        f0[~f0_uv] = f0[~f0_uv] * 2 ** (float(pitch_adjust) / 12)
        
        # intonation curve
        if intonation != 1.0:
            f0[~f0_uv] = f0[~f0_uv] * intonation ** (((f0[~f0_uv] - f0_min)/(f0_max - f0_min))*(float(f0_max) - intonation_base) / float(f0_max))

        # extract volume
        volume_extractor = VolumeExtractor(hop_size, self.args.data.volume_window_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)

        # extract units
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        units = self.units_encoder.encode(audio_t, sample_rate, hop_size)

        # spk_id or spk_mix_dict
        if self.spk_info is None:
            if use_spk_mix:
                spk_id = torch.LongTensor(np.array([int(k) for k in spk_mix_dict.keys()])).to(self.device)
                spk_mix = torch.tensor([[[float(v) for v in spk_mix_dict.values()]]]).transpose(-1, 0).to(self.device)
            else:
                spk_id = torch.LongTensor(np.array([spk_id])).to(self.device)
                spk_mix = torch.tensor([[[1.]]]).to(self.device)
        else:
            if use_spk_mix:
                spk_id = torch.stack([self.spk_embeds[str(k)]['spk_embed'] for k in spk_mix_dict.keys()]).to(self.device)
                spk_mix = torch.tensor([[[float(v) for v in spk_mix_dict.values()]]]).transpose(-1, 0).to(self.device)
            else:
                spk_id = self.spk_embeds.get(str(spk_id))
                if spk_id is None:
                    spk_id = list(self.spk_embeds.values())[0]
                spk_id = spk_id['spk_embed'].unsqueeze(0)
                spk_mix = torch.tensor([[[1.]]]).to(self.device)

        # forward and return the output
        with torch.no_grad():
            output = self.model(units, f0, volume, spk_id=spk_id, spk_mix=spk_mix)
            output *= mask
            output_sample_rate = self.args.data.sampling_rate

            output = output.squeeze()
            return output, output_sample_rate


class Config:
    def __init__(self) -> None:
        self.samplerate = 44100  # Hz
        # self.block_time = 0.3  # s
        self.block_time = 64  # frames
        self.sub_block_size = 32
        self.f_pitch_change: float = 0.0  # float(request_form.get("fPitchChange", 0))
        self.f_intonation: float = 1.0
        self.f_intonation_base: float = 200.0
        self.spk_id = 1  # 默认说话人。
        self.spk_mix_dict = None  # {1:0.5, 2:0.5} 表示1号说话人和2号说话人的音色按照0.5:0.5的比例混合
        self.use_vocoder_based_enhancer = True
        self.use_phase_vocoder = False
        self.checkpoint_path = ''
        self.threhold = -45
        self.crossfade_time = 0.04
        self.extra_time = 1.5
        self.select_pitch_extractor = 'harvest'  # F0预测器["parselmouth", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
        self.use_spk_mix = False
        self.sounddevices = ['', '']

    def save(self, path):
        with open(path + '\\config.pkl', 'wb') as f:
            pickle.dump(vars(self), f)

    def load(self, path) -> bool:
        try:
            with open(path + '\\config.pkl', 'rb') as f:
                self.update(pickle.load(f))
            return True
        except:
            print('config.pkl does not exist')
            return False
    
    def update(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)


class GUI:
    def __init__(self) -> None:
        self.config = Config()
        self.block_frame = 0
        self.crossfade_frame = 0
        self.sola_search_frame = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.svc_model: SvcDDSP = SvcDDSP()
        self.fade_in_window: np.ndarray = None  # crossfade计算用numpy数组
        self.fade_out_window: np.ndarray = None  # crossfade计算用numpy数组
        self.input_wav: np.ndarray = None  # 输入音频规范化后的保存地址
        self.output_wav: np.ndarray = None  # 输出音频规范化后的保存地址
        self.sola_buffer: torch.Tensor = None  # 保存上一个output的crossfade
        self.f0_mode_list = ["dio", "harvest", "crepe", "rmvpe", "fcpe"]  # F0预测器
        self.f_safe_prefix_pad_length: float = 0.0
        self.resample_kernel = {}
        self.stream = None
        self.input_devices = None
        self.output_devices = None
        self.input_devices_indices = None 
        self.output_devices_indices = None
        self.update_devices()
        self.default_input_device = self.input_devices[self.input_devices_indices.index(sd.default.device[0])]
        self.default_output_device = self.output_devices[self.output_devices_indices.index(sd.default.device[1])]
        self.launcher()  # start

    def launcher(self):
        '''窗口加载'''
        sg.theme('DarkAmber')  # 设置主题
        # 界面布局
        layout = [
            [sg.Frame(layout=[
                [sg.Input(key='sg_model', default_text='models\\pretrained\\mnp-svc\\vctk-full\\pytorch_model.bin'),
                 sg.FileBrowse(i18n('选择模型文件'), key='choose_model')]
            ], title=i18n('模型：.pt格式(自动识别同目录下config.yaml)')),
                sg.Frame(layout=[
                    [sg.Text(i18n('选择配置文件所在目录')), sg.Input(key='config_file_dir', default_text='exp'),
                     sg.FolderBrowse(i18n('打开文件夹'), key='choose_config')],
                    [sg.Button(i18n('读取配置文件'), key='load_config'), sg.Button(i18n('保存配置文件'), key='save_config')]
                ], title=i18n('快速配置文件'))
            ],
            [sg.Frame(layout=[
                [sg.Text(i18n("输入设备")),
                 sg.Combo(self.input_devices, key='sg_input_device', default_value=self.default_input_device,
                          enable_events=True)],
                [sg.Text(i18n("输出设备")),
                 sg.Combo(self.output_devices, key='sg_output_device', default_value=self.default_output_device,
                          enable_events=True)]
            ], title=i18n('音频设备'))
            ],
            [sg.Frame(layout=[
                [sg.Text(i18n("说话人id")), sg.Input(key='spk_id', default_text='1'), sg.Text("", key='spk_name')],
                [sg.Text(i18n("响应阈值")),
                 sg.Slider(range=(-60, 0), orientation='h', key='threhold', resolution=1, default_value=-50,
                           enable_events=True)],
                [sg.Text(i18n("变调")),
                 sg.Slider(range=(-24, 24), orientation='h', key='pitch', resolution=1, default_value=0,
                           enable_events=True)],
                [sg.Text(i18n('抑扬')),
                 sg.Slider(range=(0.0, 3.0), orientation='h', key='intonation', resolution=0.01, default_value=1.0,
                           enable_events=True)],
                [sg.Text(i18n('抑扬基(Hz)')),
                 sg.Slider(range=(50, 400), orientation='h', key='intonation_base', resolution=1, default_value=200,
                           enable_events=True)],
                [sg.Text(i18n("采样率")), sg.Input(key='samplerate', default_text='44100')],
                [sg.Checkbox(text=i18n('启用捏音色功能'), default=False, key='spk_mix', enable_events=True),
                 sg.Button(i18n("设置混合音色"), key='set_spk_mix')]
            ], title=i18n('普通设置')),
                sg.Frame(layout=[
                    [sg.Text(i18n("音频切分大小")),
                    #  sg.Slider(range=(0.05, 3.0), orientation='h', key='block', resolution=0.01, default_value=0.3,
                     sg.Slider(range=(32, 320), orientation='h', key='block', resolution=1, default_value=50,
                               enable_events=True)],
                    [sg.Text(i18n("音频切分子分割大小")),
                     sg.Slider(range=(1, 32), orientation='h', key='block_div', resolution=1, default_value=1,
                               enable_events=True)],
                    [sg.Text(i18n("交叉淡化时长")),
                    #  sg.Slider(range=(0.0, 0.15), orientation='h', key='crossfade', resolution=0.005,
                    #            default_value=0.04, enable_events=True)],
                                # default_value=0.0, enable_events=True)],
                     sg.Slider(range=(0.0, 0.5), orientation='h', key='crossfade', resolution=0.005,
                               default_value=0.03, enable_events=True)],
                    [sg.Text(i18n("额外推理时长")),
                    #  sg.Slider(range=(0.05, 5), orientation='h', key='extra', resolution=0.01, default_value=2.0,
                     sg.Slider(range=(0.0, 5.), orientation='h', key='extra', resolution=0.01, default_value=0.8,
                               enable_events=True)],
                    [sg.Text(i18n("f0预测模式")),
                     sg.Combo(values=self.f0_mode_list, key='f0_mode', default_value=self.f0_mode_list[-1],
                              enable_events=True)],
                    [sg.Checkbox(text=i18n('启用相位声码器'), default=False, key='use_phase_vocoder', enable_events=True)]
                ], title=i18n('性能设置')),
            ],
            [sg.Button(i18n("开始音频转换"), key="start_vc"), sg.Button(i18n("停止音频转换"), key="stop_vc"),
             sg.Text(i18n('推理所用时间(ms):')), sg.Text('0', key='infer_time')]
        ]

        # 创造窗口
        self.window = sg.Window('MNP-SVC - GUI', layout, finalize=True)
        self.window['spk_id'].bind('<Return>', '')
        self.window['samplerate'].bind('<Return>', '')
        self.event_handler()

    def event_handler(self):
        '''事件处理'''
        global flag_vc
        while True:  # 事件处理循环
            event, values = self.window.read()
            print('event: ' + event)
            if event == sg.WINDOW_CLOSED:  # 如果用户关闭窗口
                flag_vc = False
                exit()
            elif event == "start_vc" and not flag_vc:
                # preload model
                self.device = self.svc_model.update_model(values['sg_model'])   # read values{} is not good practice but for avoid circulate ref.
                # set values 和界面布局layout顺序一一对应
                self.set_values(values)
                print('block_time:' + str(self.config.block_time))
                print('crossfade_time:' + str(self.config.crossfade_time))
                print("extra_time:" + str(self.config.extra_time))
                print("samplerate:" + str(self.config.samplerate))
                print("prefix_pad_length:" + str(self.f_safe_prefix_pad_length))
                print("mix_mode:" + str(self.config.spk_mix_dict))
                print("enhancer:" + str(self.config.use_vocoder_based_enhancer))
                print('using_cuda:' + str(torch.cuda.is_available()))
                self.start_vc()
            elif event == 'spk_id':
                self.update_spk(values['spk_id'])
            elif event == 'threhold':
                self.config.threhold = values['threhold']
            elif event == 'pitch':
                self.config.f_pitch_change = values['pitch']
            elif event == 'intonation':
                self.config.f_intonation = values['intonation']
            elif event == 'intonation_base':
                self.config.f_intonation_base = values['intonation_base']
            elif event == 'spk_mix':
                self.config.use_spk_mix = values['spk_mix']
            elif event == 'set_spk_mix':
                spk_mix = sg.popup_get_text(message='示例：1:0.3,2:0.5,3:0.2', title="设置混合音色，支持多人")
                if spk_mix != None:
                    self.config.spk_mix_dict = eval("{" + spk_mix.replace('，', ',').replace('：', ':') + "}")
            elif event == 'f0_mode':
                self.config.select_pitch_extractor = values['f0_mode']
                self.svc_model.update_pitch_extractor(values['f0_mode'], self.config.samplerate)
            elif event == 'use_phase_vocoder':
                self.config.use_phase_vocoder = values['use_phase_vocoder']
            elif event == 'load_config' and not flag_vc:
                if self.config.load(values['config_file_dir']):
                    self.update_values()
            elif event == 'save_config' and not flag_vc:
                self.set_values(values)
                self.config.save(values['config_file_dir'])
            elif event != 'start_vc' and flag_vc:
                self.stop_stream()

    def set_values(self, values):
        self.set_devices(values["sg_input_device"], values['sg_output_device'])
        self.config.sounddevices = [values["sg_input_device"], values['sg_output_device']]
        self.config.checkpoint_path = values['sg_model']
        self.config.spk_id = int(values['spk_id'])
        self.config.threhold = values['threhold']
        self.config.f_pitch_change = values['pitch']
        self.config.f_intonation = values['intonation']
        self.config.f_intonation_base = values['intonation_base']
        self.config.samplerate = int(values['samplerate'])
        # self.config.block_time = float(values['block'])
        self.config.block_time = int(values['block'])
        self.config.crossfade_time = float(values['crossfade'])
        self.config.extra_time = float(values['extra'])
        self.config.select_pitch_extractor = values['f0_mode']
        self.config.use_phase_vocoder = values['use_phase_vocoder']
        self.config.use_spk_mix = values['spk_mix']
        self.config.sub_block_size = int(values['block_div'])
        self.block_frame = int(self.config.block_time * self.svc_model.args.data.block_size + 0.5)
        # self.callback_blocksize = max(self.block_frame//16, self.svc_model.args.data.block_size)
        # self.callback_blocksize = max(self.block_frame//16, 1)
        self.callback_blocksize = max(self.block_frame//self.config.sub_block_size, 1)
        # self.crossfade_frame = int(self.config.crossfade_time * self.config.samplerate)
        self.crossfade_frame = int(self.config.crossfade_time * self.block_frame + 0.5)
        self.sola_search_frame = int(0.005 * self.config.samplerate)
        self.last_delay_frame = int(0.01 * self.config.samplerate)
        # self.last_delay_frame = 1
        self.extra_frame = int(self.config.extra_time * self.config.samplerate)
        self.input_frame = max(
            self.block_frame + self.crossfade_frame + self.sola_search_frame + 2 * self.last_delay_frame,
            self.block_frame + self.extra_frame)
        self.f_safe_prefix_pad_length = self.config.extra_time - self.config.crossfade_time - 0.005 - 0.01

    def update_values(self):
        self.window['sg_model'].update(self.config.checkpoint_path)
        self.window['sg_input_device'].update(self.config.sounddevices[0])
        self.window['sg_output_device'].update(self.config.sounddevices[1])
        self.window['spk_id'].update(self.config.spk_id)
        self.window['threhold'].update(self.config.threhold)
        self.window['pitch'].update(self.config.f_pitch_change)
        self.window['intonation'].update(self.config.f_intonation)
        self.window['intonation_base'].update(self.config.f_intonation_base)
        self.window['samplerate'].update(self.config.samplerate)
        self.window['spk_mix'].update(self.config.use_spk_mix)
        self.window['block'].update(self.config.block_time)
        self.window['block_div'].update(self.config.sub_block_size)
        self.window['crossfade'].update(self.config.crossfade_time)
        self.window['extra'].update(self.config.extra_time)
        self.window['f0_mode'].update(self.config.select_pitch_extractor)
        
    def update_spk(self, spk_id):
        self.config.spk_id = int(spk_id)
        if self.svc_model.spk_embeds is not None:
            if str(spk_id) not in self.svc_model.spk_embeds.keys():
                self.config.spk_id = int(list(self.svc_model.spk_embeds.keys())[0])
            self.window['spk_name'].update(self.svc_model.spk_embeds[str(self.config.spk_id)]['spk_name'])
        else:
            self.window['spk_name'].update(str(self.config.spk_id))

    def start_vc(self):
        '''开始音频转换'''
        torch.cuda.empty_cache()
        self.device = self.svc_model.update_model(self.config.checkpoint_path)
        self.update_spk(self.config.spk_id)
        self.svc_model.update_pitch_extractor(self.config.select_pitch_extractor, self.config.samplerate)
        self.input_wav = np.zeros(self.input_frame, dtype='float32')
        if self.crossfade_frame > 0:
            self.fade_in_window = torch.sin(
                np.pi * torch.arange(0, 1, 1 / self.crossfade_frame, device=self.device) / 2) ** 2
            self.fade_out_window = 1 - self.fade_in_window
            self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.device)
        else:
            self.sola_search_frame = 0
            self.last_delay_frame = 0
            self.fade_in_window = 0
            self.fade_out_window = 1
            self.sola_buffer = None
        # self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.device)
        # self.fade_in_window = torch.sin(
        #     np.pi * torch.arange(0, 1, 1 / self.crossfade_frame, device=self.device) / 2) ** 2
        # self.fade_out_window = 1 - self.fade_in_window
        self.start_stream()

    def start_stream(self):
        global flag_vc
        if not flag_vc:
            flag_vc = True
            self.stream = sd.Stream(
                channels=2,
                callback=self.audio_callback,
                # blocksize=self.block_frame,
                # blocksize=max(self.block_frame//8, 32),
                blocksize=self.callback_blocksize,
                latency='low',
                samplerate=self.config.samplerate,
                dtype="float32")
            self.stream.start()

    def stop_stream(self):
        global flag_vc
        if flag_vc:
            flag_vc = False
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                
    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        '''
        音频处理
        '''
        start_time = time.perf_counter()
        # print("\nStarting callback")
        
        block_size = frames
        
        self.input_wav[:] = np.roll(self.input_wav, -block_size)
        self.input_wav[-block_size:] = librosa.to_mono(indata.T)

        # infer
        _audio, _model_sr = self.svc_model.infer(
            self.input_wav,
            self.config.samplerate,
            spk_id=self.config.spk_id,
            threhold=self.config.threhold,
            pitch_adjust=self.config.f_pitch_change,
            use_spk_mix=self.config.use_spk_mix,
            spk_mix_dict=self.config.spk_mix_dict,
            pitch_extractor_type=self.config.select_pitch_extractor,
            intonation=self.config.f_intonation,
            intonation_base=self.config.f_intonation_base,
            safe_prefix_pad_length=self.f_safe_prefix_pad_length,
        )

        # debug sola
        '''
        _audio, _model_sr = self.input_wav, self.config.samplerate
        rs = int(np.random.uniform(-200,200))
        print('debug_random_shift: ' + str(rs))
        _audio = np.roll(_audio, rs)
        _audio = torch.from_numpy(_audio).to(self.device)
        '''

        if _model_sr != self.config.samplerate:
            key_str = str(_model_sr) + '_' + str(self.config.samplerate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(_model_sr, self.config.samplerate,
                                                         lowpass_filter_width=128).to(self.device)
            _audio = self.resample_kernel[key_str](_audio)
        # temp_wav = _audio[
        #            - self.callback_blocksize - self.crossfade_frame - self.sola_search_frame - self.last_delay_frame: - self.last_delay_frame]
        temp_wav = _audio[
                   - block_size - self.crossfade_frame - self.sola_search_frame - self.last_delay_frame: - self.last_delay_frame]
        # temp_wav = _audio[:self.block_frame + self.crossfade_frame + self.sola_search_frame]
        # temp_wav = _audio[self.input_wav.shape[0] - self.block_frame - self.crossfade_frame - self.sola_search_frame:]
        
        # print(_audio.shape, temp_wav.shape, outdata.shape, self.crossfade_frame)
        
        if self.sola_buffer is not None:
            # sola shift
            # if False:
            conv_input = temp_wav[None, None, : self.crossfade_frame + self.sola_search_frame]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(
                F.conv1d(conv_input ** 2, torch.ones(1, 1, self.crossfade_frame, device=self.device)) + 1e-8)
            sola_shift = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
            # else:
            #     sola_shift = 0
            # temp_wav = temp_wav[sola_shift: sola_shift + self.block_frame + self.crossfade_frame]
            # temp_wav = temp_wav[sola_shift: sola_shift + self.callback_blocksize + self.crossfade_frame]
            temp_wav = temp_wav[sola_shift: sola_shift + block_size + self.crossfade_frame]
            print(f'\rsola_shift: {int(sola_shift):4}', end="")
        else:
            print(f'\rsola_shift: {0:4}', end="")

        # phase vocoder
        if self.config.use_phase_vocoder:
            temp_wav[: self.crossfade_frame] = phase_vocoder(
                self.sola_buffer,
                temp_wav[: self.crossfade_frame],
                self.fade_out_window,
                self.fade_in_window)
        elif self.crossfade_frame > 0:
            temp_wav[: self.crossfade_frame] *= self.fade_in_window
            temp_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window
        
        if self.crossfade_frame > 0:
            self.sola_buffer = temp_wav[- self.crossfade_frame:]

            outdata[:] = temp_wav[: - self.crossfade_frame, None].repeat(1, 2).cpu().numpy()
        else:
            # outdata[:] = _audio[: self.block_frame, None].repeat(1, 2).cpu().numpy()
            outdata[:] = _audio[: block_size, None].repeat(1, 2).cpu().numpy()
        end_time = time.perf_counter()
        print(f' infer_time: {end_time - start_time:.5f}', end="")
        if flag_vc:
            self.window['infer_time'].update(int((end_time - start_time) * 1000))

    def update_devices(self):
        '''获取设备列表'''
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        self.output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        self.input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
        self.output_devices_indices = [
            d["index"] for d in devices if d["max_output_channels"] > 0
        ]

    def set_devices(self, input_device, output_device):
        '''设置输出设备'''
        sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
        sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
        print("input device:" + str(sd.default.device[0]) + ":" + str(input_device))
        print("output device:" + str(sd.default.device[1]) + ":" + str(output_device))


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="path to the model file",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    parser.add_argument(
        "-id",
        "--spk_id",
        type=int,
        required=False,
        default=1,
        help="speaker id (for multi-speaker model) | default: 1",
    )
    parser.add_argument(
        "-semb",
        "--spk_embed",
        type=str,
        required=False,
        default="None",
        help="speaker embed .npz file (for multi-speaker with spk_embed_encoder model) | default: None",
    )
    parser.add_argument(
        "-mix",
        "--spk_mix_dict",
        type=str,
        required=False,
        default="None",
        help="mix-speaker dictionary (for multi-speaker model) | default: None",
    )
    parser.add_argument(
        "-intb",
        "--intonation_base",
        type=float,
        required=False,
        default=220.0,
        help="base freq of intonation changed | default: 220.0",
    )
    parser.add_argument(
        "-into",
        "--intonation",
        type=float,
        required=False,
        default=1.0,
        help="intonation changed (above 1.0 for exciter, below for calmer) | default: 1.0",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=int,
        required=False,
        default=0,
        help="key changed (number of semitones) | default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='rmvpe',
        help="pitch extrator type: dio, harvest, crepe, fcpe, rmvpe (default)",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=float,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=float,
        required=False,
        default=1200,
        help="max f0 (Hz) | default: 1200",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=float,
        required=False,
        default=-45,
        help="response threhold (dB) | default: -45",
    )
    # parser.add_argument(
    #     "-bt",
    #     "--block_time",
    #     type=float,
    #     default=0.3,
    # )
    parser.add_argument(
        "-bf",
        "--block_frame",
        type=int,
        default=64,
    )
    # parser.add_argument(
    #     "-ct",
    #     "--crossfade_time",
    #     type=float,
    #     # default=0.04,
    #     default=0.,
    # )
    parser.add_argument(
        "-cf",
        "--crossfade_frame",
        type=int,
        # default=0.04,
        default=16,
    )
    parser.add_argument(
        "-et",
        "--extra_time",
        type=float,
        # default=1.5,
        # default=0.,
        default=0.8,
    )
    parser.add_argument(
       "-pb" ,
       "--phase_vocoder",
       action="store_true",
    )
    return parser.parse_args(args=args, namespace=namespace)


class OfflineRenderer(GUI):
    def __init__(self, cmd, sr=44100) -> None:
        self.cmd = cmd
        self.config = Config()
        self.block_frame = 0
        self.crossfade_frame = 0
        self.sola_search_frame = 0
        if cmd.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = cmd.device
        self.svc_model: SvcDDSP = SvcDDSP()
        self.fade_in_window: np.ndarray = None  # crossfade计算用numpy数组
        self.fade_out_window: np.ndarray = None  # crossfade计算用numpy数组
        self.input_wav: np.ndarray = None  # 输入音频规范化后的保存地址
        self.output_wav: np.ndarray = None  # 输出音频规范化后的保存地址
        self.sola_buffer: torch.Tensor = None  # 保存上一个output的crossfade
        self.f0_mode_list = ["dio", "harvest", "crepe", "rmvpe", "fcpe"]  # F0预测器
        self.f_safe_prefix_pad_length: float = 0.0
        self.resample_kernel = {}
        self.stream = None
        
        self.config.checkpoint_path = cmd.model_path
        self.config.spk_id = cmd.spk_id
        self.config.threhold = cmd.threhold
        self.config.f_pitch_change = cmd.key
        self.config.f_intonation = cmd.intonation
        self.config.f_intonation_base = cmd.intonation_base
        self.config.samplerate = sr
        # self.config.block_time = cmd.block_time
        
        self.config.extra_time = cmd.extra_time
        self.config.select_pitch_extractor = cmd.pitch_extractor
        self.config.use_phase_vocoder = cmd.phase_vocoder
        self.config.use_spk_mix = cmd.spk_mix_dict != "None"
        self.config.spk_mix_dict = eval("{" + cmd.spk_mix_dict.replace('，', ',').replace('：', ':') + "}")
        
        self.sola_search_frame = int(0.01 * self.config.samplerate)
        self.last_delay_frame = int(0.02 * self.config.samplerate)
        self.extra_frame = int(self.config.extra_time * self.config.samplerate)
        
        self.device = self.svc_model.update_model(self.config.checkpoint_path)
        self.config.block_time = cmd.block_frame*self.svc_model.args.data.block_size / self.config.samplerate
        self.block_frame = int(self.config.block_time * self.config.samplerate + 0.5)
        
        self.config.crossfade_time = cmd.crossfade_frame * self.svc_model.args.data.block_size / self.config.samplerate
        self.crossfade_frame = int(self.config.crossfade_time * self.config.samplerate + 0.5)
        self.input_frame = max(
            self.block_frame + self.crossfade_frame + self.sola_search_frame + 2 * self.last_delay_frame,
            self.block_frame + self.extra_frame)
        
        self.f_safe_prefix_pad_length = self.config.extra_time - self.config.crossfade_time - 0.01 - 0.02
        self.svc_model.update_pitch_extractor(self.config.select_pitch_extractor, self.config.samplerate)
        self.input_wav = np.zeros(self.input_frame, dtype='float32')
        if self.crossfade_frame > 0:
            self.fade_in_window = torch.sin(
                np.pi * torch.arange(0, 1, 1 / self.crossfade_frame, device=self.device) / 2) ** 2
            self.fade_out_window = 1 - self.fade_in_window
            self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.device)
        else:
            self.sola_search_frame = 0
            self.last_delay_frame = 0
        # self.fade_in_window = torch.sin(
        #     np.pi * torch.arange(0, 1, 1 / self.crossfade_frame, device=self.device) / 2) ** 2
        # self.fade_out_window = 1 - self.fade_in_window
        
    def render(self, indata: np.ndarray, outdata: np.ndarray):
        super().audio_callback(indata, outdata, 0, 0, None)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        # launch GUI
        i18n = gui_locale.I18nAuto()
        gui = GUI()
    else:
        # offline rendering
        import soundfile as sf
        cmd = parse_args()
        info = sf.info(cmd.input)
        renderer = OfflineRenderer(cmd, info.samplerate)
        
        # blocksize = int(cmd.block_time * info.samplerate)
        blocksize = int(renderer.block_frame)
        print(f"blocksize: {blocksize}")
        buffer_frames = int(info.frames//blocksize + 1) * blocksize
        result = np.zeros((1, buffer_frames, 2), dtype=np.float64)
        for idx, block in enumerate(sf.blocks(cmd.input, blocksize=blocksize, fill_value=0.0)):
            renderer.render(block, result[:, idx*blocksize:(idx+1)*blocksize])
            
        delayed_frames = renderer.sola_search_frame + renderer.crossfade_frame + renderer.last_delay_frame
        sf.write(cmd.output, result[:, delayed_frames:info.frames+delayed_frames, 0].squeeze(0), info.samplerate)
    
