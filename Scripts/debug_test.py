import serialNo_class

settings = serialNo_class.serialNo_experiment()
settings.without_contrast()
# settings.only_t1w_with_contrast()
# settings.all_modals_with_contrast()
# settings.vgg16_parameters()
settings.vgg16_and_mlp_parameters()
settings.NUM_EPOCHS_mlp = 1
settings.NUM_EPOCHS_vgg16 = 1
# ex = serialNo_class.pretrained_VGG16(settings)
ex = serialNo_class.VGG16_and_MLP(settings)
# ex.dataset_A_experiment()
ex.dataset_B_and_C_experiment()