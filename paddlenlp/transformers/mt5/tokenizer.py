import os
import unicodedata
from shutil import copyfile

from paddle.utils import try_import
from .. import PretrainedTokenizer


__all__ = ['MT5Tokenizer']

class MT5Tokenizer(PretrainedTokenizer):
    resource_files_names = {"vocab_file": "vocab.txt"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "t5-small": "https://huggingface.co/t5-small/resolve/main/spiece.model",
            "t5-base": "https://huggingface.co/t5-base/resolve/main/spiece.model",
            "t5-large": "https://huggingface.co/t5-large/resolve/main/spiece.model",
            "t5-3b": "https://huggingface.co/t5-3b/resolve/main/spiece.model",
            "t5-11b": "https://huggingface.co/t5-11b/resolve/main/spiece.model",
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

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ) :
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids id already formatted with special tokens for the model."
                )
            return super().get_special_tokens_mask(token_ids_0=token_ids_0,
                                                   token_ids_1=token_ids_1,
                                                   already_has_special_tokens=already_has_special_tokens)
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids):
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0, token_ids_1=None
    ):
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=None
    ):
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text: str):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            token = f"<extra_id_{self.vocab_size - 1 - index}>"
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode_pieces(current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode_pieces(current_sub_tokens)
        return out_string.strip()

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Copy vocab file to {out_vocab_file}")

        return (out_vocab_file,)
