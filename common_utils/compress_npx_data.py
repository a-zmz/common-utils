import spikeinterface.extractors as se
from wavpack_numcodecs import WavPack

# TODO nov8 cannot install wavpack_numcodecs
path = "~/running_data/npx/interim/"
name = "20210310_az_HFR20/catgt_20210310_az_HFR20_g1/20210310_az_HFR20_g1_tcat.imec0.ap.bin"

recording = se.read_spikeglx(path+name)

compressor = WavPack(level=3)

recording.save(
    format='zarr',
    folder=path+"test.zarr",
    compressor=compressor,
    n_job=10,
)
