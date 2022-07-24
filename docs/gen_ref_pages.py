"""
Generate the code reference pages.
This script is created by referring to the URL : https://mkdocstrings.github.io/recipes/.
"""

from pathlib import Path

import mkdocs_gen_files

SORCE_DIR = "jijbench"

nav = mkdocs_gen_files.Nav()

for path in sorted(Path(SORCE_DIR).rglob("*.py")):

    module_py_file_path = path.relative_to(SORCE_DIR)
    module_path = module_py_file_path.with_suffix("")
    doc_path = module_py_file_path.with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    elif parts[-1][0] == "_":
        # "_" prefix file is secret for user.
        continue

    if len(parts) == 0:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = SORCE_DIR + "." + ".".join(parts)
        fd.write(f"::: {identifier}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)  #

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
