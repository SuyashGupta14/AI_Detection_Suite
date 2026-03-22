/* model2.js — Satellite Image Classifier */

const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const previewBox = document.getElementById('previewBox');
const previewImg = document.getElementById('previewImg');
const changeBtn = document.getElementById('changeBtn');
const idleState = document.getElementById('idleState');
const loadingState = document.getElementById('loadingState');
const resultState = document.getElementById('resultState');
const resetBtn = document.getElementById('resetBtn');

// Guard if model didn't load
if (!uploadBox) { console.warn('Satellite UI missing — model may not be loaded'); }

if (uploadBox) {

    uploadBox.addEventListener('click', () => imageInput.click());

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#4caf82';
    });
    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.borderColor = '#3a3a4a';
    });
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) handleImage(f);
        else alert('Please drop an image file (JPG, PNG, WEBP)');
    });

    imageInput.addEventListener('change', () => {
        if (imageInput.files[0]) handleImage(imageInput.files[0]);
    });

    changeBtn.addEventListener('click', () => imageInput.click());
    resetBtn.addEventListener('click', resetUI);
}

// ── Main flow ─────────────────────────────────────────────────────
function handleImage(file) {
    // Show preview immediately
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadBox.style.display = 'none';
        previewBox.style.display = 'flex';
    };
    reader.readAsDataURL(file);

    setPanel('loading');

    const fd = new FormData();
    fd.append('image', file);

    fetch('/predict_satellite', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                setPanel('idle');
                return;
            }
            showResult(data);
        })
        .catch(() => {
            alert('Server error — check terminal');
            setPanel('idle');
        });
}

// ── Show result ───────────────────────────────────────────────────
function showResult(data) {
    // Top card
    document.getElementById('resultEmoji').textContent = data.emoji;
    document.getElementById('resultLabel').textContent = data.label;
    document.getElementById('resultDesc').textContent = data.desc;
    document.getElementById('resultConf').textContent = data.confidence + '%';

    // Confidence badge
    const badge = document.getElementById('confBadge');
    badge.textContent = data.conf_level + ' confidence';
    badge.className = 'conf-badge ' + data.conf_level.toLowerCase();

    // Color result card border/bg
    const card = document.getElementById('resultCard');
    card.style.borderColor = data.color;
    card.style.background = hexToRgba(data.color, 0.07);

    // Build probability bars
    const container = document.getElementById('satBars');
    container.innerHTML = '';

    data.probabilities.forEach(item => {
        const isTop = item.class === data.class;
        const row = document.createElement('div');
        row.className = 'sat-bar-row';
        row.innerHTML = `
      <span class="sat-bar-emoji">${item.emoji}</span>
      <span class="sat-bar-label" style="${isTop ? 'color:#fff;font-weight:700' : ''}">${item.label}</span>
      <div class="sat-bar-track">
        <div class="sat-bar-fill" id="bar_${item.class}" style="background:${item.color};width:0%"></div>
      </div>
      <span class="sat-bar-pct" style="${isTop ? 'color:#fff;font-weight:700' : ''}">${item.pct}%</span>
    `;
        container.appendChild(row);
    });

    setPanel('result');

    // Animate bars after DOM renders
    setTimeout(() => {
        data.probabilities.forEach(item => {
            const bar = document.getElementById('bar_' + item.class);
            if (bar) bar.style.width = item.pct + '%';
        });
    }, 80);
}

// ── Panel switcher ────────────────────────────────────────────────
function setPanel(state) {
    idleState.style.display = state === 'idle' ? 'flex' : 'none';
    loadingState.style.display = state === 'loading' ? 'flex' : 'none';
    resultState.style.display = state === 'result' ? 'flex' : 'none';
}

// ── Reset ─────────────────────────────────────────────────────────
function resetUI() {
    previewBox.style.display = 'none';
    uploadBox.style.display = 'flex';
    imageInput.value = '';
    previewImg.src = '';
    setPanel('idle');
}

// ── Utility ───────────────────────────────────────────────────────
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}