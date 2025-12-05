# GitHub Pages Setup

This repository uses GitHub Actions to automatically build and deploy the Quarto website to GitHub Pages.

## Initial Setup (One-time)

1. **Enable GitHub Pages in your repository:**
   - Go to your repository → **Settings** → **Pages**
   - Under **Source**, select: **GitHub Actions**
   - Save

2. **Push the workflow file:**
   - The `.github/workflows/publish.yml` file is already created
   - Commit and push to your repository

3. **First deployment:**
   - After pushing, GitHub Actions will automatically run
   - Check the **Actions** tab to see the workflow progress
   - Once complete, your site will be available at:
     `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`

## How It Works

- **Automatic**: Every time you push to `main` or `master`, the workflow:
  1. Converts `README.md` to `webpage.qmd`
  2. Renders the Quarto document to HTML
  3. Deploys to GitHub Pages

- **Manual trigger**: You can also manually trigger the workflow from the **Actions** tab

## Local Development

To preview locally before pushing:

```bash
cd webpage
./render.sh
# Open webpage.html in your browser
```

## Troubleshooting

- **Workflow fails**: Check the **Actions** tab for error messages
- **Site not updating**: Make sure GitHub Pages is set to use **GitHub Actions** as the source
- **Missing files**: Ensure all files in `webpage/` are committed (except `webpage.html` and `site_libs/` which are generated)

