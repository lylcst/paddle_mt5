import os
import unicodedata
from shutil import copyfile

from paddle.utils import try_import
from .. import PretrainedTokenizer


__all__ = ['MT5Tokenizer']

class MT5Tokenizer(PretrainedTokenizer):
  resource_files_names = {"vocab_file": "vocab.txt"} 
  pretrained_resource_files_map = {
        "vocab_file": {'mt5-base-v1': ''}
  }
  pretrained_init_configuration = {
    "mt5-base-v1": {
        "do_lower_case": True
    }
  }
  def __init__(self,
             vocab_file,
             do_lower_case=True,
             unk_token="[UNK]",
             sep_token="[SEP]",
             pad_token="[PAD]",
             cls_token="[CLS]",
             mask_token="[MASK]"):

    if not os.path.isfile(vocab_file):
        raise ValueError(
            "Can't find a vocabulary file at path '{}'. To load the "
            "vocabulary from a pretrained model please use "
            "`tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            .format(vocab_file))
    self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(
        vocab=self.vocab, unk_token=unk_token)
  
