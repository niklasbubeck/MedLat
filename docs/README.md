# MedTok Project Page

This folder is the source for the [GitHub Pages](https://pages.github.com/) project site.  
**Template:** [Nerfies](https://github.com/nerfies/nerfies.github.io) (CC BY-SA 4.0).

## Setup Instructions

1. **Enable GitHub Pages** (one-time):
   - Go to your repo: `https://github.com/YOUR_USERNAME/MedTok`
   - Click **Settings** → **Pages** (under "Code and automation")
   - Under "Build and deployment" → "Source": select **Deploy from a branch**
   - Branch: `main` (or your default branch)
   - Folder: **/docs**
   - Click **Save**

2. **Update links** in `index.html`:
   - Replace `YOUR_USERNAME` with your GitHub username (3 places)
   - Add Paper and arXiv URLs when published
   - Add author names and affiliations

3. **Deploy**: Push changes to `main`. The site will be live at:
   ```
   https://YOUR_USERNAME.github.io/MedTok/
   ```

## Structure

- `index.html` — Main page (hero, teaser, carousel, abstract, BibTeX)
- `static/` — CSS (Bulma), JS, images, figures from `results/figures/`
