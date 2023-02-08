import hydra
import hydra.utils
from omegaconf import OmegaConf


@hydra.main(config_path=".", config_name="hydra", version_base=None)
def main(cfg: OmegaConf):
    print(cfg)
    print(hydra.utils.instantiate(cfg))


if __name__ == "__main__":
    main()
