import locale
'''
本地化方式如下所示
'''

LANGUAGE_LIST = ['zh_CN', 'en_US', 'ja_JP']
LANGUAGE_ALL = {
    'zh_CN': {
        'SUPER': 'END',
        'LANGUAGE': 'zh_CN',
        '选择模型文件': '选择模型文件',
        '模型：.pt格式(自动识别同目录下config.yaml)': '模型：.pt格式(自动识别同目录下config.yaml)',
        '选择配置文件所在目录': '选择配置文件所在目录',
        '打开文件夹': '打开文件夹',
        '读取配置文件': '读取配置文件',
        '保存配置文件': '保存配置文件',
        '快速配置文件': '快速配置文件',
        '输入设备': '输入设备',
        '输出设备': '输出设备',
        '音频设备': '音频设备',
        '说话人id': '说话人id',
        '响应阈值': '响应阈值',
        '变调': '变调',
        '采样率': '采样率',
        '启用捏音色功能': '启用捏音色功能',
        '设置混合音色': '设置混合音色',
        '普通设置': '普通设置',
        '音频切分大小': '音频切分大小',
        '交叉淡化时长': '交叉淡化时长',
        '额外推理时长': '额外推理时长',
        'f0预测模式': 'f0预测模式',
        '启用增强器': '启用增强器',
        '启用相位声码器': '启用相位声码器',
        '性能设置': '性能设置',
        '开始音频转换': '开始音频转换',
        '停止音频转换': '停止音频转换',
        '推理所用时间(ms):': '推理所用时间(ms):',
        '抑扬': '抑扬',
        '抑扬基(Hz)': '抑扬基(Hz)',
        '音频切分子分割大小': '音频切分子分割大小',
    },
    'en_US': {
        'SUPER': 'zh_CN',
        'LANGUAGE': 'en_US',
        '选择模型文件': 'Select Model File',
        '模型：.pt格式(自动识别同目录下config.yaml)': 'Model：.pt format(Auto ust config.yaml in here)',
        '选择配置文件所在目录': 'Select the configuration file directory',
        '打开文件夹': 'Open folder',
        '读取配置文件': 'Read config file',
        '保存配置文件': 'Save config file',
        '快速配置文件': 'Fast config file',
        '输入设备': 'Input device',
        '输出设备': 'Output device',
        '音频设备': 'Audio devices',
        '说话人id': 'Speaker ID',
        '响应阈值': 'Response threshold',
        '变调': 'Pitch',
        '采样率': 'Sampling rate',
        '启用捏音色功能': 'Enable Mix Speaker',
        '设置混合音色': 'Mix Speaker',
        '普通设置': 'Normal Settings',
        '音频切分大小': 'Segmentation size',
        '交叉淡化时长': 'Cross fade duration',
        '额外推理时长': 'Extra inference time',
        'f0预测模式': 'f0Extractor',
        '启用增强器': 'Enable Enhancer',
        '启用相位声码器': 'Enable Phase Vocoder',
        '性能设置': 'Performance settings',
        '开始音频转换': 'Start conversion',
        '停止音频转换': 'Stop conversion',
        '推理所用时间(ms):': 'Inference time(ms):',
        '抑扬': 'Intonation',
        '抑扬基(Hz)': 'Intonation base(Hz)',
        '音频切分子分割大小': 'Subsegmentation size',
    },
    'ja_JP': {
        'SUPER': 'zh_CN',
        'LANGUAGE': 'ja_JP',
        '选择模型文件': 'モデルを選択',
        '模型：.pt格式(自动识别同目录下config.yaml)': 'モデル：.pt形式（同じディレクトリにあるconfig.yamlを自動認識します）',
        '选择配置文件所在目录': '設定ファイルを選択',
        '打开文件夹': 'フォルダを開く',
        '读取配置文件': '設定ファイルを読み込む',
        '保存配置文件': '設定ファイルを保存',
        '快速配置文件': '設定プロファイル',
        '输入设备': '入力デバイス',
        '输出设备': '出力デバイス',
        '音频设备': '音声デバイス',
        '说话人id': '話者ID',
        '响应阈值': '応答時の閾値',
        '变调': '音程',
        '采样率': 'サンプリングレート',
        '启用捏音色功能': 'ミキシングを有効化',
        '设置混合音色': 'ミキシング',
        '普通设置': '通常設定',
        '音频切分大小': 'セグメンテーションのサイズ',
        '交叉淡化时长': 'クロスフェードの間隔',
        '额外推理时长': '追加推論時間',
        'f0预测模式': 'f0予測モデル',
        '启用增强器': 'Enhancerを有効化',
        '启用相位声码器': 'フェーズボコーダを有効化',
        '性能设置': 'パフォーマンスの設定',
        '开始音频转换': '変換開始',
        '停止音频转换': '変換停止',
        '推理所用时间(ms):': '推論時間(ms):',
        '抑扬': '抑揚',
        '抑扬基(Hz)': '抑揚の基準(Hz)',
        '音频切分子分割大小': 'サブセグメンテーションのサイズ',
    }
}


class I18nAuto:
    def __init__(self, language=None):
        self.language_list = LANGUAGE_LIST
        self.language_all = LANGUAGE_ALL
        self.language_map = {}
        if language is None:
            language = 'auto'
        if language == 'auto':
            language = locale.getdefaultlocale()[0]
            if language not in self.language_list:
                language = 'zh_CN'
        self.language = language
        super_language_list = []
        while self.language_all[language]['SUPER'] != 'END':
            super_language_list.append(language)
            language = self.language_all[language]['SUPER']
        super_language_list.append('zh_CN')
        super_language_list.reverse()
        for _lang in super_language_list:
            self.read_language(self.language_all[_lang])

    def read_language(self, lang_dict: dict):
        for _key in lang_dict.keys():
            self.language_map[_key] = lang_dict[_key]

    def __call__(self, key):
        return self.language_map[key]
