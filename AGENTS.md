# Repository Guidelines

## Project Structure & Module Organization
- Core site lives in `src/`: content in `paper.mdx`, pages in `pages/`, reusable UI in `components/`, styles in `styles/`, helpers in `lib/`, and shared types in `types/`.
- Place images and media in `src/assets/`; anything in `public/` is served as-is at the site root. `dist/` is build output—do not edit manually.

## Build, Test, and Development Commands
- `npm run dev` (alias `npm start`): Astro dev server at http://localhost:4321 with hot reload.
- `npm run build`: Runs `astro check` (type/syntax validation) then production build to `dist/`.
- `npm run preview`: Serves the built site locally for final review.
- `npm run lint` / `npm run lint:fix`: ESLint for Astro/TS/React/Markdown/JSON/CSS; `:fix` applies safe fixes.
- `npm run format` / `npm run format:fix`: Prettier dry-run or write with Astro and Tailwind plugins.

## Coding Style & Naming Conventions
- TypeScript, Astro, and React/TSX are preferred; keep components PascalCase (e.g., `HighlightedSection.astro`, `Comparison.tsx`).
- Follow Prettier defaults (2-space indent, 100ish line wraps); run `npm run format:fix` before review. Keep imports sorted logically.
- Keep MDX concise; favor repository components (e.g., `Figure`, `Video`, `ModelViewer`) instead of raw HTML for consistency.
- CSS uses Tailwind v4; prefer utility classes and keep custom rules in `src/styles/global.css` when needed.

## Testing Guidelines
- No dedicated automated test suite; rely on `astro check`, `npm run build`, and `npm run preview` to catch regressions.
- When adding components, verify both light and dark modes and responsive states (desktop/tablet/mobile). Include representative screenshots or clips in the PR if behavior is visual.

## Commit & Pull Request Guidelines
- Commit messages should be short, imperative, and scoped (e.g., `feat: add carousel slide captions`, `chore: format assets`).
- PRs should describe what changed, why, and how to validate (commands run, pages to view). Link related issues or paper sections if relevant.
- Screenshots or short screen recordings are expected for UI changes. Note any follow-up tasks or TODOs explicitly in the description.

## Content & Asset Tips
- Optimize images before adding; prefer modern formats (WebP/AVIF). Keep large media in `public/` if it should bypass the Astro pipeline.
- For new MDX content, co-locate figures in `src/assets/` and reference them with relative paths to keep the project portable.
