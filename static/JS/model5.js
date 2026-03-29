/* model5.js - Fake Job Posting Detection */

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.fraud-tab-content').forEach(t => t.style.display = 'none');
        document.getElementById('tab-' + btn.dataset.tab).style.display = 'block';
    });
});

function analyzeFakeJobText() {
    const text = document.getElementById('jobText').value.trim();
    if (!text) {
        alert('Please paste job posting text first.');
        return;
    }

    setJobPanel('loading');

    fetch('/predict_fake_job', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                setJobPanel('idle');
                return;
            }
            showJobResult(data);
        })
        .catch(() => {
            alert('Server error - check terminal');
            setJobPanel('idle');
        });
}

function showJobResult(data) {
    const card = document.getElementById('jobVerdictCard');
    card.style.borderColor = data.color;
    card.style.background = hexToRgba(data.color, 0.08);

    document.getElementById('jobVerdictEmoji').textContent = data.emoji;
    document.getElementById('jobVerdictLabel').textContent = data.prediction;
    document.getElementById('jobVerdictConf').textContent = data.confidence + '%';
    document.getElementById('jobVerdictModel').textContent = `${data.model_used} · ${data.vectorizer}`;

    const badge = document.getElementById('jobVerdictBadge');
    badge.textContent = data.conf_level + ' confidence';
    badge.className = 'conf-badge ' + data.conf_level.toLowerCase();

    setJobPanel('result');

    setTimeout(() => {
        document.getElementById('jobRealBar').style.width = data.real_prob + '%';
        document.getElementById('jobFakeBar').style.width = data.fake_prob + '%';
        document.getElementById('jobRealPct').textContent = data.real_prob + '%';
        document.getElementById('jobFakePct').textContent = data.fake_prob + '%';
    }, 80);
}

function setJobPanel(state) {
    const idle = document.getElementById('jobIdle');
    const loading = document.getElementById('jobLoading');
    const result = document.getElementById('jobResult');
    if (!idle || !loading || !result) return;

    idle.style.display = state === 'idle' ? 'flex' : 'none';
    loading.style.display = state === 'loading' ? 'flex' : 'none';
    result.style.display = state === 'result' ? 'flex' : 'none';
}

function loadJobSample() {
    const sample =
        'Immediate Hiring! Work from home and earn $5,000 weekly. No experience required. ' +
        'Send your personal details and processing fee to start today.';
    document.getElementById('jobText').value = sample;
    setJobPanel('idle');
}

function clearJobText() {
    document.getElementById('jobText').value = '';
    setJobPanel('idle');
}

const jobCsvUploadBox = document.getElementById('jobCsvUploadBox');
const jobCsvInput = document.getElementById('jobCsvInput');

if (jobCsvUploadBox) {
    jobCsvUploadBox.addEventListener('click', () => jobCsvInput.click());

    jobCsvUploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        jobCsvUploadBox.style.borderColor = '#f5c76f';
    });

    jobCsvUploadBox.addEventListener('dragleave', () => {
        jobCsvUploadBox.style.borderColor = '#3a3a4a';
    });

    jobCsvUploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        const f = e.dataTransfer.files[0];
        if (f) handleJobCSV(f);
    });

    jobCsvInput.addEventListener('change', () => {
        if (jobCsvInput.files[0]) handleJobCSV(jobCsvInput.files[0]);
    });
}

function handleJobCSV(file) {
    jobCsvUploadBox.style.display = 'none';
    document.getElementById('jobBulkLoading').style.display = 'flex';
    document.getElementById('jobBulkResults').style.display = 'none';

    const fd = new FormData();
    fd.append('file', file);

    fetch('/predict_fake_job_csv', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            document.getElementById('jobBulkLoading').style.display = 'none';
            if (data.error) {
                alert('Error: ' + data.error);
                resetJobBulk();
                return;
            }
            showJobBulk(data);
        })
        .catch(() => {
            document.getElementById('jobBulkLoading').style.display = 'none';
            alert('Server error - check terminal');
            resetJobBulk();
        });
}

function showJobBulk(data) {
    document.getElementById('jobSummTotal').textContent = data.total;
    document.getElementById('jobSummReal').textContent = data.real_count;
    document.getElementById('jobSummFake').textContent = data.fake_count;
    document.getElementById('jobSummPct').textContent = data.fake_pct + '%';
    document.getElementById('jobBulkModelUsed').textContent = data.model_used;
    document.getElementById('jobBulkVectorizer').textContent = data.vectorizer;

    const tbody = document.getElementById('jobBulkTableBody');
    tbody.innerHTML = '';

    data.rows.forEach(row => {
        const tr = document.createElement('tr');
        tr.className = row.is_fake ? 'fraud-row' : '';
        tr.innerHTML = `
      <td>${row.row}</td>
      <td style="font-size:12px;max-width:480px;">${escapeHtml(row.preview)}</td>
      <td>${row.emoji} ${row.prediction}</td>
      <td>${row.confidence}%</td>
    `;
        tbody.appendChild(tr);
    });

    document.getElementById('jobBulkResults').style.display = 'block';
}

function resetJobBulk() {
    if (!jobCsvUploadBox) return;
    jobCsvUploadBox.style.display = 'block';
    document.getElementById('jobBulkLoading').style.display = 'none';
    document.getElementById('jobBulkResults').style.display = 'none';
    jobCsvInput.value = '';
}

function escapeHtml(s) {
    return String(s)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}
