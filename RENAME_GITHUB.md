# Rename GitHub Repository to MedLat

The project has been renamed from **MedTok** to **MedLat** locally. To complete the rename on GitHub:

## Steps

1. **Rename the repository on GitHub**
   - Go to https://github.com/niklasbubeck/MedTok
   - Click **Settings** (repo settings, not account)
   - In "Repository name", change `MedTok` to `MedLat`
   - Click **Rename**

2. **Update your local remote** (if needed)
   ```bash
   git remote set-url origin https://github.com/niklasbubeck/MedLat.git
   ```

3. **Update GitHub Pages URL**
   - After rename, your project page will be at: `https://niklasbubeck.github.io/MedLat/`
   - GitHub automatically redirects old URLs, but update any external links.

4. **Re-clone or re-add** (for collaborators)
   - Old clone URLs will redirect to the new name
   - Collaborators may need to update their remotes

## Note

GitHub will set up redirects from the old repo URL to the new one. Existing clones will continue to work, but it's good practice to update the remote URL.
