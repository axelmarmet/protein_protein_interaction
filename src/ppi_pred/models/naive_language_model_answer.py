import torch
from torch import nn

from torch import Tensor

from ppi_pred.constants import MAX_LENGTH
from ppi_pred.models.encodings.positional_encoding import PositionalEncoding
from ppi_pred.models.encodings.segment_encoding import SegmentEncoding


class NaivePPILanguageModel(nn.Module):

    def __init__(self, embedding_dim):
        super(NaivePPILanguageModel, self).__init__()

        self.embedding_dim = embedding_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=1024,
        )
        encoder_norm = nn.LayerNorm(embedding_dim)

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            6,
            encoder_norm
        )

        self.classifier = nn.Linear(embedding_dim, 1)

        self.positional_encoding = PositionalEncoding(
            embedding_dim, dropout=0.1, maxlen=MAX_LENGTH)
        self.fst_seq_encoding = SegmentEncoding(embedding_dim, dropout=0.1)
        self.snd_seq_encoding = SegmentEncoding(embedding_dim, dropout=0.1)

        self.cls_embedding = nn.parameter.Parameter(
            torch.randn(embedding_dim), requires_grad=True)
        self.sep_embedding = nn.parameter.Parameter(
            torch.randn(embedding_dim), requires_grad=True)

    def forward(self, fst_seq: Tensor, snd_seq: Tensor):
        """ forward iteration
        Args:
            fst_seq (Tensor): shape  `seq_1_len x batch_dim x embedding_dim`
            snd_seq (Tensor): shape `seq_2_len x batch_dim x embedding_dim`
        """

        # validate a bit the inputs
        fst_seq_len, batch_dim, embedding_dim = fst_seq.shape
        snd_seq_len, batch_dim_, embedding_dim_ = snd_seq.shape

        assert batch_dim == batch_dim_, "batch dim of the two sequences are not the same"
        assert embedding_dim == embedding_dim_, "embedding dim of the two sequences are not the same"

        fst_segment = torch.concat([
            self.cls_embedding.tile((1, batch_dim, 1)),
            fst_seq,
            self.sep_embedding.tile((1, batch_dim, 1))
        ], 0)

        snd_segment = torch.concat([
            snd_seq,
            self.sep_embedding.tile((1, batch_dim, 1))
        ], 0)

        fst_segment = self.fst_seq_encoding(fst_segment)
        snd_segment = self.snd_seq_encoding(snd_segment)

        encoder_input = torch.cat([fst_segment, snd_segment], 0)
        encoder_input = self.positional_encoding(encoder_input)

        cls_token_embedding = self.encoder(encoder_input)[0,:]

        return torch.sigmoid(self.classifier(cls_token_embedding))

EMBEDDING_DIM = 64
BATCH_DIM = 16

SEQ_1_LEN = 9
SEQ_2_LEN = 23

fst_seq = torch.randn((SEQ_1_LEN, BATCH_DIM, EMBEDDING_DIM))
snd_seq = torch.randn((SEQ_2_LEN, BATCH_DIM, EMBEDDING_DIM))

model = NaivePPILanguageModel(EMBEDDING_DIM)

res = model(fst_seq, snd_seq)

print(res)