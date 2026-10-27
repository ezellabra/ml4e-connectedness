"""Run the full pipeline: data checks, estimation and comparison, robustness, figures,
optional forecasting. Usage: python run_all.py"""
from src import data, figures, forecast, pipeline


def main() -> None:
    data.main()
    pipeline.main()
    figures.main()
    forecast.main()


if __name__ == "__main__":
    main()
