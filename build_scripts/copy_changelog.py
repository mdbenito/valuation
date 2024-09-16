import logging
import os
from pathlib import Path

import mkdocs.plugins

logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent
docs_dir = root_dir / "docs"
changelog_file = root_dir / "CHANGELOG.md"
target_filepath = docs_dir / "changelog.md"


@mkdocs.plugins.event_priority(100)
def on_pre_build(config):
    logger.info("Temporarily copying changelog to docs directory")
    try:
        if os.path.getmtime(changelog_file) <= os.path.getmtime(target_filepath):
            logger.info(
                f"Changelog '{os.fspath(changelog_file)}' hasn't been updated, skipping."
            )
            return
    except FileNotFoundError:
        pass
    logger.info(
        f"Creating symbolic link for '{os.fspath(changelog_file)}' "
        f"at '{os.fspath(target_filepath)}'"
    )
    target_filepath.symlink_to(changelog_file)

    logger.info("Finished copying changelog to docs directory")


@mkdocs.plugins.event_priority(-100)
def on_shutdown():
    logger.info("Removing temporary changelog in docs directory")
    target_filepath.unlink()
