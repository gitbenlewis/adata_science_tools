import importlib.util
import re
import struct
import sys
import unittest
from html import escape, unescape
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"

manifest_spec = importlib.util.spec_from_file_location(
    "_documentation_gallery_manifest",
    REPO_ROOT / "example_plotting_gallery" / "manifest.py",
)
manifest_module = importlib.util.module_from_spec(manifest_spec)
sys.modules[manifest_spec.name] = manifest_module
manifest_spec.loader.exec_module(manifest_module)
RENDERER_MANIFEST = manifest_module.RENDERER_MANIFEST


class DocumentationTests(unittest.TestCase):
    def test_readme_setup_is_safe_and_uses_current_plot_api(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertLess(
            readme.index("git clone https://github.com/gitbenlewis/adata_science_tools.git"),
            readme.index("cd adata_science_tools"),
        )
        self.assertLess(
            readme.index("cd adata_science_tools"),
            readme.index("conda env create -f config/env_not_base.yaml -n not_base"),
        )
        self.assertNotIn("conda remove -n not_base --all", readme)
        self.assertIn("adtl.datapoints_effect_panels_column()", readme)
        self.assertNotIn("adtl.barh_l2fc_dotplot()", readme)

    def test_local_documentation_links_resolve(self):
        markdown_paths = [REPO_ROOT / "README.md", *sorted(DOCS_DIR.glob("*.md"))]
        markdown_link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)")
        html_target_pattern = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")
        missing_targets = []

        for markdown_path in markdown_paths:
            markdown = markdown_path.read_text(encoding="utf-8")
            targets = markdown_link_pattern.findall(markdown)
            targets.extend(html_target_pattern.findall(markdown))
            for raw_target in targets:
                target = unescape(raw_target)
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target_path = (
                    markdown_path.parent / unquote(parsed.path)
                ).resolve()
                if not target_path.exists():
                    missing_targets.append(
                        f"{markdown_path.relative_to(REPO_ROOT)} -> {target}"
                    )

        self.assertEqual(missing_targets, [])

    def test_docs_index_lists_every_module_page(self):
        docs_index = (DOCS_DIR / "README.md").read_text(encoding="utf-8")
        module_pages = sorted(path.name for path in DOCS_DIR.glob("_*.md"))

        for module_page in module_pages:
            with self.subTest(page=module_page):
                self.assertIn(f"]({module_page})", docs_index)

    def test_gallery_catalog_matches_manifest_and_has_deep_links(self):
        catalog = (DOCS_DIR / "plotting_gallery.md").read_text(encoding="utf-8")
        renderer_doc_overrides = {
            "paired_datapoints": "_paired_datapoints.md",
        }

        for renderer in RENDERER_MANIFEST:
            renderer_anchor = f'renderer-{renderer.name}'
            expected_doc = renderer_doc_overrides.get(
                renderer.name,
                f"{renderer.module.rsplit('.', 1)[-1]}.md",
            )
            with self.subTest(renderer=renderer.name):
                self.assertEqual(
                    catalog.count(f'<a id="{renderer_anchor}"></a>'),
                    1,
                )
                self.assertIn(
                    f'<a href="{expected_doc}"><code>{renderer.name}</code></a>',
                    catalog,
                )
                self.assertIn(
                    f'<a href="#{renderer_anchor}">Permalink</a>',
                    catalog,
                )
                if renderer.replacement is not None:
                    self.assertIn(
                        f'href="#renderer-{renderer.replacement}">'
                        f'<code>{renderer.replacement}</code></a>',
                        catalog,
                    )

            for case in renderer.cases:
                asset_path = f"assets/plotting_gallery/{case.asset}"
                png = (DOCS_DIR / asset_path).read_bytes()
                image_width, image_height = struct.unpack(">II", png[16:24])
                aspect_ratio = image_width / image_height
                display_width = (
                    700
                    if aspect_ratio >= 2.2
                    else 520
                    if aspect_ratio >= 1.25
                    else 400
                )
                with self.subTest(renderer=renderer.name, case=case.case_id):
                    self.assertEqual(
                        catalog.count(f'<img src="{asset_path}"'),
                        1,
                    )
                    self.assertEqual(
                        catalog.count(f'<a href="{asset_path}">'),
                        1,
                    )
                    self.assertIn(
                        f"<code>{case.case_id}</code> — {case.title}",
                        catalog,
                    )
                    self.assertIn(
                        f'<img src="{asset_path}" '
                        f'alt="{escape(case.title, quote=True)}" '
                        f'width="{display_width}">',
                        catalog,
                    )

    def test_documentation_images_have_alt_text(self):
        markdown_paths = [REPO_ROOT / "README.md", *sorted(DOCS_DIR.glob("*.md"))]
        html_image_pattern = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
        html_alt_pattern = re.compile(r"\balt=[\"']([^\"']+)[\"']", re.IGNORECASE)
        markdown_image_pattern = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
        missing_alt_text = []

        for markdown_path in markdown_paths:
            markdown = markdown_path.read_text(encoding="utf-8")
            for image_tag in html_image_pattern.findall(markdown):
                alt_match = html_alt_pattern.search(image_tag)
                if alt_match is None or not alt_match.group(1).strip():
                    missing_alt_text.append(
                        f"{markdown_path.relative_to(REPO_ROOT)}: {image_tag}"
                    )
            for alt_text in markdown_image_pattern.findall(markdown):
                if not alt_text.strip():
                    missing_alt_text.append(
                        f"{markdown_path.relative_to(REPO_ROOT)}: empty Markdown alt"
                    )

        self.assertEqual(missing_alt_text, [])


if __name__ == "__main__":
    unittest.main()
