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
  def __init__(
          self,
          vocab_file,
          sentencepiece_model_file,
          eos_token="</s>",
          unk_token="<unk>",
          pad_token="<pad>",
          extra_ids=100,
          additional_special_tokens=None,
          **kwargs
      ):
          # Add extra_ids to the special token list
          if extra_ids > 0 and additional_special_tokens is None:
              additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
          elif extra_ids > 0 and additional_special_tokens is not None:
              # Check that we have the right number of extra_id special tokens
              extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
              if extra_tokens != extra_ids:
                  raise ValueError(
                      f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. "
                      "In this case the additional_special_tokens must include the extra_ids tokens"
                  )

          self.vocab_file = vocab_file
          self._extra_ids = extra_ids
          spm = try_import("sentencepiece")
          self.sp_model = spm.SentencePieceProcessor()
          self.sp_model.Load(sentencepiece_model_file)
    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

