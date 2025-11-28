"""Simple demo showing how to instantiate models via the central registry."""

from src import available_models, get_model, MODEL_REGISTRY
from src.continuous import AutoencoderKL_f4
from src.discrete import VQModel
from src.modules import Encoder, Decoder
from src.discrete.quantizer import SimVQ

def main() -> None:
    print("Available token models:", available_models(""))

    encoder = Encoder(img_size=128, dims=2)
    decoder = Decoder(img_size=128, dims=2)
    quantizer = SimVQ(in_channels=3, n_e=8192, e_dim=3)
    vq_model = VQModel(encoder=encoder, decoder=decoder, quantizer=quantizer)
    print(f"VQ model codebook size: {vq_model.quantizer.n_e}")



if __name__ == "__main__":
    main()

