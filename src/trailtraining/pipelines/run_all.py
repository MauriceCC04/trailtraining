import logging
import os
import shutil
from typing import Optional

from trailtraining import config
from trailtraining.data import combine as combine_jsons
from trailtraining.pipelines import strava as strava_pipeline

log = logging.getLogger(__name__)


def _clean_directory(directory: str) -> None:
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        p = os.path.join(directory, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except FileNotFoundError:
            pass


def _detect_provider(explicit: Optional[str] = None) -> str:
    """
    Decide which wellness provider to use.

    Priority:
      1) explicit arg (CLI --wellness-provider)
      2) env TRAILTRAINING_WELLNESS_PROVIDER or WELLNESS_PROVIDER
      3) auto-detect based on available credentials (Intervals preferred if configured)
    """
    if explicit:
        v = explicit.strip().lower()
    else:
        env_v = (os.getenv("TRAILTRAINING_WELLNESS_PROVIDER") or "").strip() or (
            os.getenv("WELLNESS_PROVIDER") or ""
        ).strip()
        v = env_v.lower() if env_v else "auto"

    if v in {"garmin", "intervals"}:
        return v

    # auto-detect
    if (config.INTERVALS_API_KEY or "").strip():
        return "intervals"
    if (config.GARMIN_EMAIL or "").strip() and (config.GARMIN_PASSWORD or "").strip():
        return "garmin"

    # fallback: Intervals (least external dependency)
    return "intervals"


def main(
    *,
    clean: bool = False,
    clean_processing: bool = False,
    clean_prompting: bool = False,
    wellness_provider: Optional[str] = None,
) -> None:
    """
    Run full pipeline:
      - (Garmin OR Intervals) wellness fetch → Strava → Combine

    IMPORTANT:
    - By default we DO NOT wipe processing/, so Strava incremental state (processing/strava_meta.json) is preserved.
    - Use clean / clean_processing to force a full Strava refetch.
    """
    config.ensure_directories()

    if clean:
        clean_processing = True
        clean_prompting = True

    if clean_processing:
        _clean_directory(config.PROCESSING_DIRECTORY)
    if clean_prompting:
        _clean_directory(config.PROMPTING_DIRECTORY)

    provider = _detect_provider(wellness_provider)
    log.info("Selected wellness provider: %s", provider)

    if provider == "garmin":
        # Lazy import so Intervals users don't need GarminDb installed.
        from trailtraining.pipelines import garmin as garmin_pipeline

        # garmin_pipeline.main() now handles per-profile GarminDb HOME isolation + config activation
        garmin_pipeline.main()
    else:
        # Lazy import so Garmin users don't need Intervals dependencies.
        from trailtraining.pipelines import intervals as intervals_pipeline

        intervals_pipeline.main()

    # run the strava pipeline (incremental if processing/strava_meta.json exists)
    strava_pipeline.main()
    # combine the jsons in the prompting directory
    combine_jsons.main()

    log.info("All pipelines completed successfully.")


if __name__ == "__main__":
    main()
