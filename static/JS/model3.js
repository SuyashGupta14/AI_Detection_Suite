/* model3.js — Credit Card Fraud Detection */

// ── Active model ──────────────────────────────────────────────────
let activeModel = document.querySelector('.model-btn.active')?.dataset.key || 'xgboost';

document.querySelectorAll('.model-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.model-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeModel = btn.dataset.key;
    });
});

// ── Tab switching ─────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.fraud-tab-content').forEach(t => t.style.display = 'none');
        document.getElementById('tab-' + btn.dataset.tab).style.display = 'block';
    });
});


// ══════════════════════════════════════════════════════════════════
// TAB 1 — Single Transaction
// ══════════════════════════════════════════════════════════════════

function analyzeSingle() {
    // Read datetime-local (convert to match training format)
    const dtRaw = document.getElementById('f_trans_date_trans_time').value;
    const dt = dtRaw ? dtRaw.replace('T', ' ') + ':00' : '2020-06-21 12:14:25';

    const payload = {
        model: activeModel,
        trans_date_trans_time: dt,
        amt: parseFloat(document.getElementById('f_amt').value) || 0,
        category: document.getElementById('f_category').value,
        gender: document.getElementById('f_gender').value,
        lat: parseFloat(document.getElementById('f_lat').value) || 0,
        long: parseFloat(document.getElementById('f_long').value) || 0,
        merch_lat: parseFloat(document.getElementById('f_merch_lat').value) || 0,
        merch_long: parseFloat(document.getElementById('f_merch_long').value) || 0,
        dob: document.getElementById('f_dob').value,
        city_pop: parseFloat(document.getElementById('f_city_pop').value) || 0,
        state: document.getElementById('f_state').value,
        zip: document.getElementById('f_zip').value,
    };

    setSinglePanel('loading');

    fetch('/predict_fraud_single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    })
        .then(r => r.json())
        .then(data => {
            if (data.error) { alert('Error: ' + data.error); setSinglePanel('idle'); return; }
            showSingleResult(data);
        })
        .catch(() => { alert('Server error — check terminal'); setSinglePanel('idle'); });
}

function showSingleResult(data) {
    const card = document.getElementById('verdictCard');
    card.style.borderColor = data.color;
    card.style.background = hexToRgba(data.color, 0.08);

    document.getElementById('verdictEmoji').textContent = data.emoji;
    document.getElementById('verdictLabel').textContent = data.prediction;
    document.getElementById('verdictConf').textContent = data.confidence + '%';
    document.getElementById('verdictModel').textContent =
        data.model_used + '  ·  Accuracy: ' + data.model_acc + '  ·  ROC-AUC: ' + data.model_roc;

    const badge = document.getElementById('verdictBadge');
    badge.textContent = data.conf_level + ' confidence';
    badge.className = 'conf-badge ' + data.conf_level.toLowerCase();

    setSinglePanel('result');

    setTimeout(() => {
        document.getElementById('legitBar').style.width = data.legit_prob + '%';
        document.getElementById('fraudBar').style.width = data.fraud_prob + '%';
        document.getElementById('legitPct').textContent = data.legit_prob + '%';
        document.getElementById('fraudPct').textContent = data.fraud_prob + '%';
    }, 80);
}

function setSinglePanel(state) {
    document.getElementById('singleIdle').style.display = state === 'idle' ? 'flex' : 'none';
    document.getElementById('singleLoading').style.display = state === 'loading' ? 'flex' : 'none';
    document.getElementById('singleResult').style.display = state === 'result' ? 'flex' : 'none';
}

function resetSingle() {
    setSinglePanel('idle');
    document.getElementById('legitBar').style.width = '0%';
    document.getElementById('fraudBar').style.width = '0%';
}

// ── Sample data matching your actual dataset ──────────────────────
function fillSample(isFraud) {
    // Legitimate transaction (from your row 0)
    const legit = {
        dt: '2020-06-21T12:14:25', amt: 2.86, category: 'personal_care', gender: 'M',
        lat: 33.9659, long: -80.9355, merch_lat: 33.986, merch_long: -81.200,
        dob: '1968-03-19', city_pop: 333497, state: 'SC', zip: 29209,
    };
    // Fraud-like transaction (high amount, far merchant, unusual hour)
    const fraud = {
        dt: '2020-06-21T02:30:00', amt: 1842.99, category: 'shopping_net', gender: 'F',
        lat: 40.6729, long: -73.5365, merch_lat: 33.986, merch_long: -81.200,
        dob: '1990-01-17', city_pop: 302, state: 'NY', zip: 11710,
    };

    const s = isFraud ? fraud : legit;
    document.getElementById('f_trans_date_trans_time').value = s.dt;
    document.getElementById('f_amt').value = s.amt;
    document.getElementById('f_category').value = s.category;
    document.getElementById('f_gender').value = s.gender;
    document.getElementById('f_lat').value = s.lat;
    document.getElementById('f_long').value = s.long;
    document.getElementById('f_merch_lat').value = s.merch_lat;
    document.getElementById('f_merch_long').value = s.merch_long;
    document.getElementById('f_dob').value = s.dob;
    document.getElementById('f_city_pop').value = s.city_pop;
    document.getElementById('f_state').value = s.state;
    document.getElementById('f_zip').value = s.zip;

    setSinglePanel('idle');
}

function clearForm() {
    document.getElementById('f_amt').value = '';
    document.getElementById('f_city_pop').value = '';
    document.getElementById('f_lat').value = '';
    document.getElementById('f_long').value = '';
    document.getElementById('f_merch_lat').value = '';
    document.getElementById('f_merch_long').value = '';
    setSinglePanel('idle');
}


// ══════════════════════════════════════════════════════════════════
// TAB 2 — Bulk CSV
// ══════════════════════════════════════════════════════════════════

const csvUploadBox = document.getElementById('csvUploadBox');
const csvInput = document.getElementById('csvInput');

if (csvUploadBox) {
    csvUploadBox.addEventListener('click', () => csvInput.click());
    csvUploadBox.addEventListener('dragover', (e) => {
        e.preventDefault(); csvUploadBox.style.borderColor = '#f38ba8';
    });
    csvUploadBox.addEventListener('dragleave', () => {
        csvUploadBox.style.borderColor = '#3a3a4a';
    });
    csvUploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files[0]) handleCSV(e.dataTransfer.files[0]);
    });
    csvInput.addEventListener('change', () => {
        if (csvInput.files[0]) handleCSV(csvInput.files[0]);
    });
}

function handleCSV(file) {
    csvUploadBox.style.display = 'none';
    document.getElementById('bulkLoading').style.display = 'flex';
    document.getElementById('bulkResults').style.display = 'none';

    const fd = new FormData();
    fd.append('file', file);
    fd.append('model', activeModel);

    fetch('/predict_fraud_csv', { method: 'POST', body: fd })
        .then(r => r.json())
        .then(data => {
            document.getElementById('bulkLoading').style.display = 'none';
            if (data.error) { alert('Error: ' + data.error); resetBulk(); return; }
            showBulkResults(data);
        })
        .catch(() => {
            document.getElementById('bulkLoading').style.display = 'none';
            alert('Server error'); resetBulk();
        });
}

function showBulkResults(data) {
    document.getElementById('summTotal').textContent = data.total;
    document.getElementById('summLegit').textContent = data.legit_count;
    document.getElementById('summFraud').textContent = data.fraud_count;
    document.getElementById('summPct').textContent = data.fraud_pct + '%';
    document.getElementById('bulkModelUsed').textContent = data.model_used;
    document.getElementById('bulkModelAcc').textContent = 'Acc: ' + data.model_acc;

    const tbody = document.getElementById('bulkTableBody');
    tbody.innerHTML = '';
    data.rows.forEach(row => {
        const tr = document.createElement('tr');
        tr.className = row.is_fraud ? 'fraud-row' : '';
        tr.innerHTML = `
      <td>${row.row}</td>
      <td>$${row.amount}</td>
      <td style="font-size:11px;color:#888">${row.merchant}</td>
      <td style="font-size:11px">${row.category.replace('_', ' ')}</td>
      <td>${row.emoji} ${row.prediction}</td>
      <td>${row.confidence}%</td>
    `;
        tbody.appendChild(tr);
    });

    document.getElementById('bulkResults').style.display = 'block';
}

function resetBulk() {
    csvUploadBox.style.display = 'flex';
    document.getElementById('bulkLoading').style.display = 'none';
    document.getElementById('bulkResults').style.display = 'none';
    csvInput.value = '';
}

// ── Utility ───────────────────────────────────────────────────────
function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}