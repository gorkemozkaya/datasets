
"""Blended TR EN dataset."""

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.translate import wmt

_URL = ""
_CITATION = ""

_LANGUAGE_PAIRS = [
    ("tr", "en")
]


class BlendedTranslate(wmt.WmtTranslate):
  """Blended translation datasets for the {"tr", "en"} language pair."""

  BUILDER_CONFIGS = [
      wmt.WmtConfig(  # pylint:disable=g-complex-comprehension
          description="Blended tr-en translation task dataset.",
          url=_URL,
          citation=_CITATION,
          language_pair=("tr", "en"),
          version=tfds.core.Version("1.0.0"),
      )
  ]

  @property
  def _subsets(self):
    return {
        tfds.Split.TRAIN: [
            "blended_en_tr_train"
        ],
        tfds.Split.VALIDATION: ["blended_en_tr_dev"]
    }
