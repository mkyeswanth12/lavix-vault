const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const dir = 'dist/assets';
const html = 'dist/index.html';
const buildId = Date.now().toString(36) + crypto.randomBytes(4).toString('hex');

// A. Replace Vite's 8-char hash with the buildId in every JS filename.
//    Keeping the .js extension is critical — later steps scan dir for *.js.
const tempNames = {};
for (const f of fs.readdirSync(dir)) {
  if (!f.endsWith('.js')) continue;
  const src = path.join(dir, f);
  const tmp = f.replace(/-([a-zA-Z0-9_-]{8})\.js$/, '-' + buildId + '.js');
  fs.renameSync(src, path.join(dir, tmp));
  tempNames[f] = tmp;
}

// B. Update all cross-references from original names to buildId names.
const allTargets = [html];
for (const f of fs.readdirSync(dir)) {
  if (f.endsWith('.js')) allTargets.push(path.join(dir, f));
}
for (const fp of allTargets) {
  let content = fs.readFileSync(fp, 'utf8');
  let changed = false;
  for (const [o, n] of Object.entries(tempNames)) {
    if (content.includes(o)) {
      content = content.split(o).join(n);
      changed = true;
    }
  }
  if (changed) fs.writeFileSync(fp, content);
}

// C. Compute content hash of each buildId-named file and rename to final.
const finalNames = {};
for (const [orig, tmp] of Object.entries(tempNames)) {
  const fp = path.join(dir, tmp);
  const hash = crypto.createHash('sha256').update(fs.readFileSync(fp)).digest('hex').slice(0, 8);
  const final = tmp.replace(/-([a-zA-Z0-9_-]{8})\.js$/, '-' + hash + '.js');
  fs.renameSync(fp, path.join(dir, final));
  finalNames[tmp] = final;
}

// D. Update all cross-references from buildId names to final hash names.
for (const fp of allTargets) {
  let content = fs.readFileSync(fp, 'utf8');
  let changed = false;
  for (const [o, n] of Object.entries(finalNames)) {
    if (content.includes(o)) {
      content = content.split(o).join(n);
      changed = true;
    }
  }
  if (changed) fs.writeFileSync(fp, content);
}
