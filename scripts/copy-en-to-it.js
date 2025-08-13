const fs = require('fs');
const path = require('path');

const srcDir = path.join(__dirname, '..', 'docs');
const destDir = path.join(__dirname, '..', 'i18n', 'it', 'docusaurus-plugin-content-docs', 'current');

function copyRecursive(src, dest) {
    if (!fs.existsSync(dest)) {
        fs.mkdirSync(dest, { recursive: true });
    }

    fs.readdirSync(src).forEach(file => {
        const srcPath = path.join(src, file);
        const destPath = path.join(dest, file);

        if (fs.lstatSync(srcPath).isDirectory()) {
            copyRecursive(srcPath, destPath);
        } else if (/\.(md|mdx)$/.test(file)) {
            if (!fs.existsSync(destPath)) {
                fs.copyFileSync(srcPath, destPath);
                console.log(`Copied: ${srcPath} → ${destPath}`);
            } else {
                console.log(`Skipped (already exists): ${destPath}`);
            }
        }
    });
}

copyRecursive(srcDir, destDir);