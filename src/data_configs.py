DATA_DIR = "./data/"
DATA_CONFIGS = {
    "a9a": {"tr": "a9a", "tst": "a9a.t", "loading": "libsvm"},
    "acsincome": {
        "tr": "acsincome_data.pkl",
        "tgt": "acsincome_target.pkl",
        "loading": "pkl",
    },
    "airlines": {
        "tr": "airlines_data.pkl",
        "tgt": "airlines_target.pkl",
        "loading": "pkl",
    },
    "aspirin": {"tr": "md17_aspirin.npz", "loading": "npz"},
    "benzene": {"tr": "md17_benzene2017.npz", "loading": "npz"},
    "cadata": {"tr": "cadata", "loading": "libsvm"},
    "click_prediction": {
        "tr": "click_prediction_data.pkl",
        "tgt": "click_prediction_target.pkl",
        "loading": "pkl",
    },
    "cod_rna": {"tr": "cod-rna", "tst": "cod-rna.t", "loading": "libsvm"},
    "comet_mc": {
        "tr": "comet_mc_data.pkl",
        "tgt": "comet_mc_target.pkl",
        "loading": "pkl",
    },
    "connect_4": {"tr": "connect-4", "loading": "libsvm"},
    "covtype_binary": {"tr": "covtype.libsvm.binary.scale", "loading": "libsvm"},
    "creditcard": {
        "tr": "creditcard_data.pkl",
        "tgt": "creditcard_target.pkl",
        "loading": "pkl",
    },
    "diamonds": {
        "tr": "diamonds_data.pkl",
        "tgt": "diamonds_target.pkl",
        "loading": "pkl",
    },
    "ethanol": {"tr": "md17_ethanol.npz", "loading": "npz"},
    "higgs": {"tr": "HIGGS", "loading": "libsvm"},
    "hls4ml": {"tr": "hls4ml_data.pkl", "tgt": "hls4ml_target.pkl", "loading": "pkl"},
    "ijcnn1": {"tr": "ijcnn1.tr", "tst": "ijcnn1.t", "loading": "libsvm"},
    "jannis": {"tr": "jannis_data.pkl", "tgt": "jannis_target.pkl", "loading": "pkl"},
    "malonaldehyde": {"tr": "md17_malonaldehyde.npz", "loading": "npz"},
    "medical": {
        "tr": "medical_data.pkl",
        "tgt": "medical_target.pkl",
        "loading": "pkl",
    },
    "miniboone": {
        "tr": "miniboone_data.pkl",
        "tgt": "miniboone_target.pkl",
        "loading": "pkl",
    },
    "mnist": {"tr": "mnist_data.pkl", "tgt": "mnist_target.pkl", "loading": "pkl"},
    "naphthalene": {"tr": "md17_naphthalene.npz", "loading": "npz"},
    "phishing": {"tr": "phishing", "loading": "libsvm"},
    "qm9": {"tr": "homo.mat", "loading": "mat"},
    "santander": {
        "tr": "santander_data.pkl",
        "tgt": "santander_target.pkl",
        "loading": "pkl",
    },
    "salicylic": {"tr": "md17_salicylic.npz", "loading": "npz"},
    "sensit_vehicle": {
        "tr": "combined_scale",
        "tst": "combined_scale.t",
        "loading": "libsvm",
    },
    "sensorless": {
        "tr": "Sensorless.scale.tr",
        "tst": "Sensorless.scale.val",
        "loading": "libsvm",
    },
    "skin_nonskin": {"tr": "skin_nonskin", "loading": "libsvm"},
    "susy": {"tr": "SUSY", "loading": "libsvm"},
    "synthetic": {},
    "taxi": {"tr": "taxi-data/subsampled_data.h5py", "loading": "h5py"},
    "toluene": {"tr": "md17_toluene.npz", "loading": "npz"},
    "uracil": {"tr": "md17_uracil.npz", "loading": "npz"},
    "volkert": {
        "tr": "volkert_data.pkl",
        "tgt": "volkert_target.pkl",
        "loading": "pkl",
    },
    "w8a": {"tr": "w8a", "tst": "w8a.t", "loading": "libsvm"},
    "yearpredictionmsd": {
        "tr": "YearPredictionMSD",
        "tst": "YearPredictionMSD.t",
        "loading": "libsvm",
    },
    "yolanda": {
        "tr": "yolanda_data.pkl",
        "tgt": "yolanda_target.pkl",
        "loading": "pkl",
    },
}
DATA_KEYS = list(DATA_CONFIGS.keys())
SYNTHETIC_NTR = 10000
SYNTHETIC_NTST = 1000
SYNTHETIC_D = 10
