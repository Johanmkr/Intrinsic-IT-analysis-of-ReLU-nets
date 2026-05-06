import pathlib as pl

current_dir = pl.Path(__file__).parent.parent.resolve()

neurips_figpath = current_dir / "figures"
neurips_figpath.mkdir(parents=True, exist_ok=True)
