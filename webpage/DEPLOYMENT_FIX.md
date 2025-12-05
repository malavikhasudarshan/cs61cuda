# Fix GitHub Pages Deployment

Your site is currently showing the old version. Here's how to fix it:

## Steps to Deploy Quarto Version

### 1. Update GitHub Pages Settings

1. Go to your repository: https://github.com/malavikhasudarshan/cs61cuda
2. Click **Settings** → **Pages**
3. Under **Source**, change from whatever it's currently set to → **GitHub Actions**
4. Save

### 2. Commit and Push the Updated Workflow

The workflow has been updated to:
- Render Quarto properly
- Output `index.html` (required for GitHub Pages)
- Deploy via GitHub Actions

```bash
git add .github/workflows/publish.yml
git add webpage/
git commit -m "Update GitHub Actions workflow for Quarto deployment"
git push
```

### 3. Trigger the Workflow

After pushing:
- Go to **Actions** tab in your GitHub repo
- You should see "Publish Quarto Website" workflow running
- Wait for it to complete (usually 1-2 minutes)
- Once it shows ✅, your site will be updated

### 4. Manual Trigger (if needed)

If the workflow doesn't run automatically:
- Go to **Actions** tab
- Click "Publish Quarto Website" workflow
- Click "Run workflow" button
- Select your branch and run

## What Changed

- ✅ Workflow now renames `webpage.html` → `index.html` (required for GitHub Pages)
- ✅ Quarto config updated to output `index.html` directly
- ✅ All necessary files are in place

## Verify Deployment

After the workflow completes:
- Visit: https://malavikhasudarshan.github.io/cs61cuda/
- You should see:
  - ✅ Sidebar table of contents on the left
  - ✅ Numbered sections
  - ✅ Code folding buttons
  - ✅ CS61C-style formatting

If you still see the old version:
- Clear your browser cache (Ctrl+Shift+R or Cmd+Shift+R)
- Wait a few minutes for GitHub Pages to update
- Check the Actions tab to ensure workflow succeeded

