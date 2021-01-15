from transformers import EncoderDecoderModel

model_path = 'rubert_cased_L-12_H-768_A-12_pt'
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_path, model_path)
model.save_pretrained('pretrained_init_enc_dec')
