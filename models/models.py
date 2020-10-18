def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cGAN':
        from .cGAN_model import cGANModel
        model = cGANModel()
    elif opt.model == 'geoGAN':
        from .geoGAN_model import geoGANModel
        model = geoGANModel()
    elif opt.model == 'vae':
        opt.which_model_netG = 'resnet_6blocks'
        from .vae_model import VAEModel
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
