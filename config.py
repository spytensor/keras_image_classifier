### define global configs  ###

class DefaultConfigs(object):
    #1. string configs
    train_data = "../data/train/"
    val_data = "../data/val/"  # if exists else use train_test_split to generate val dataset
    test_data = "../data/all/traffic-sign/test/00003/"  # for competitions
    model_name = "NASNetMobile"
    weights_path = "./checkpoints/model.h5"#save weights for predict
    
    #2. numerical configs
    lr = 0.001
    epochs = 50
    num_classes = 62
    image_size = 224
    batch_size = 16
    channels = 3
    gpu = "0"
    
config = DefaultConfigs()
