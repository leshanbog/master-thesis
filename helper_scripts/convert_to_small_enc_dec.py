from transformers import EncoderDecoderModel, EncoderDecoderConfig, BertConfig, AutoModel


enc_dec_model_path = 'pretrained_init_enc_dec'
bert_model_path = 'rubert_cased_L-12_H-768_A-12_pt'


enc_config = BertConfig.from_pretrained(bert_model_path)
enc_config.num_hidden_layers = 8

enc_model = AutoModel.from_pretrained(bert_model_path, config=enc_config)
enc_model.save_pretrained('pretrained_enc_8_layers')



dec_config = BertConfig.from_pretrained(bert_model_path)
dec_config.num_hidden_layers = 6

dec_model = AutoModel.from_pretrained(bert_model_path, config=dec_config)
dec_model.save_pretrained('pretrained_dec_6_layers')
