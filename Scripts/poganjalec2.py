import os
import sys
import rf_modality_SerialNo
import VGG16_mlp_modality_SerialNo
import VGG16_modality_SerialNo
import mlp_modality_SerialNo

contrast = False
all_modal = False
b_dataset = False
c_dataset = False

rf_modality_SerialNo.ex_rf(contrast, all_modal, b_dataset, c_dataset)
VGG16_mlp_modality_SerialNo.ex_vgg_mlp(contrast, all_modal, b_dataset, c_dataset)

b_dataset = True

VGG16_modality_SerialNo.ex_vgg16(contrast, all_modal, b_dataset, c_dataset)
mlp_modality_SerialNo.ex_mlp(contrast, all_modal, b_dataset, c_dataset)
rf_modality_SerialNo.ex_rf(contrast, all_modal, b_dataset, c_dataset)
VGG16_mlp_modality_SerialNo.ex_vgg_mlp(contrast, all_modal, b_dataset, c_dataset)

b_dataset = False
c_dataset = True

VGG16_modality_SerialNo.ex_vgg16(contrast, all_modal, b_dataset, c_dataset)
mlp_modality_SerialNo.ex_mlp(contrast, all_modal, b_dataset, c_dataset)
rf_modality_SerialNo.ex_rf(contrast, all_modal, b_dataset, c_dataset)
VGG16_mlp_modality_SerialNo.ex_vgg_mlp(contrast, all_modal, b_dataset, c_dataset)

c_dataset = False
contrast = True

VGG16_modality_SerialNo.ex_vgg16(contrast, all_modal, b_dataset, c_dataset)
mlp_modality_SerialNo.ex_mlp(contrast, all_modal, b_dataset, c_dataset)
rf_modality_SerialNo.ex_rf(contrast, all_modal, b_dataset, c_dataset)
VGG16_mlp_modality_SerialNo.ex_vgg_mlp(contrast, all_modal, b_dataset, c_dataset)

contrast = False
all_modal = True

VGG16_modality_SerialNo.ex_vgg16(contrast, all_modal, b_dataset, c_dataset)
mlp_modality_SerialNo.ex_mlp(contrast, all_modal, b_dataset, c_dataset)
rf_modality_SerialNo.ex_rf(contrast, all_modal, b_dataset, c_dataset)
VGG16_mlp_modality_SerialNo.ex_vgg_mlp(contrast, all_modal, b_dataset, c_dataset)