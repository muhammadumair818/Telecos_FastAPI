// Helper for Formatting
const fmt = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });

let sessionId = null;
let storedPlots = null;
let plotsRendered = false;

// --- Markdown formatting ---
function formatMarkdown(text) {
    if (typeof marked !== 'undefined') return marked.parse(text);
    return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
}

// --- File name display (safe if element missing) ---
const fileNameSpan = document.getElementById('fileName');
if (fileNameSpan) {
    document.getElementById('fileInput').addEventListener('change', function(e) {
        fileNameSpan.textContent = e.target.files[0]?.name || 'No file chosen';
    });
} else {
    document.getElementById('fileInput').addEventListener('change', function(e) {});
}

// --- Upload & initial analysis ---
$('#uploadForm').on('submit', function(e) {
    e.preventDefault();
    const file = $('#fileInput')[0].files[0];
    if (!file) { alert('Please select a file'); return; }
    let fd = new FormData();
    fd.append('file', file);
    $('#loading').show();
    $('#results').hide();
    $.ajax({
        url: '/upload',
        type: 'POST',
        data: fd,
        processData: false,
        contentType: false,
        success: function(data) {
            sessionId = data.session_id;
            $('#loading').hide();
            $('#results').show();
            displayKPIs(data.kpis);
            storedPlots = data.plots;
            plotsRendered = false;
            if ($('.tab-btn.active').data('tab') === 'track') {
                renderPlots();
            }
            displayMLScores(data);
            $('#chatMessages').html('<div class="chat-message assistant flex"><div class="bg-gray-800 rounded-2xl px-4 py-2 max-w-[75%]">✅ Data loaded! Ask me about your tower data.</div></div>');
            $('#recommendations').hide();

            // Populate filter dropdowns (factories, shifts) from the dataset
            populateFilters();
            // Apply default filters (full dataset) to populate filtered sections
            applyFilters();
        },
        error: function(xhr) {
            $('#loading').hide();
            alert('Upload failed: ' + (xhr.responseJSON?.detail || 'Unknown error'));
        }
    });
});

// --- Original KPIs display (used for unfiltered data) ---
function displayKPIs(kpis) {
    let html = '';
    for (let [k, v] of Object.entries(kpis)) {
        let val = (typeof v === 'number') ? v.toFixed(2) : v;
        if (k.includes('Revenue') || k.includes('Cost') || k.includes('Profit')) val = '$' + val;
        if (k.includes('Utilization')) val = (v * 100).toFixed(1) + '%';
        html += `<div class="bg-black/30 backdrop-blur rounded-xl p-4 text-center border border-gray-800 hover:border-blue-500/50 transition">
                    <div class="text-xs text-gray-400 uppercase tracking-wider">${k}</div>
                    <div class="text-2xl font-bold text-white mt-1">${val}</div>
                </div>`;
    }
    $('#kpis').html(html);
}

// --- Filtered KPIs display (separate container) ---
function displayFilteredKPIs(kpis) {
    let html = '';
    for (let [k, v] of Object.entries(kpis)) {
        let val = (typeof v === 'number') ? v.toFixed(2) : v;
        if (k.includes('Revenue') || k.includes('Cost') || k.includes('Profit')) val = '$' + val;
        if (k.includes('Utilization')) val = (v * 100).toFixed(1) + '%';
        html += `<div class="bg-black/30 backdrop-blur rounded-xl p-4 text-center border border-gray-800 hover:border-blue-500/50 transition">
                    <div class="text-xs text-gray-400 uppercase tracking-wider">${k}</div>
                    <div class="text-2xl font-bold text-white mt-1">${val}</div>
                </div>`;
    }
    $('#filteredKpis').html(html);
}

// --- Render stored plots (existing) ---
function renderPlots() {
    if (!storedPlots) return;
    let container = $('#plots');
    container.empty();
    if (Object.keys(storedPlots).length === 0) {
        container.html('<div class="bg-black/30 rounded-xl p-4 text-center">No plots available for this dataset.</div>');
        return;
    }
    for (let [name, figJson] of Object.entries(storedPlots)) {
        let divId = 'plot_' + name.replace(/[^a-zA-Z0-9]/g, '_');
        container.append(`<div id="${divId}" class="plot-container bg-black/30 rounded-xl p-3"></div>`);
        try {
            let fig = JSON.parse(figJson);
            Plotly.newPlot(divId, fig.data, fig.layout, {responsive: true, displayModeBar: false});
        } catch (e) {
            console.error('Plot error:', name, e);
            $(`#${divId}`).html('<div class="p-4 text-red-400">Plot could not be rendered.</div>');
        }
    }
    plotsRendered = true;
}

// --- ML model performance (existing) ---
function displayMLScores(data) {
    let html = '<div class="bg-black/30 rounded-xl p-4"><strong class="text-blue-400">🤖 Model Performance</strong><br>';
    if (data.rev_score) html += `Revenue Model R²: ${data.rev_score.toFixed(4)}<br>`;
    if (data.cost_score) html += `Cost Model R²: ${data.cost_score.toFixed(4)}<br>`;
    if (data.clf_acc) html += `Classification Accuracy: ${(data.clf_acc * 100).toFixed(1)}%<br>`;
    html += '</div>';
    $('#mlScores').html(html);
}

// --- Revenue Prediction ---
$('#btnRev').click(function() {
    if (!sessionId) { alert('Upload data first'); return; }
    let fd = new FormData();
    fd.append('active_tenants', $('#rev_tenants').val());
    fd.append('energy_cost', $('#rev_energy').val());
    fd.append('opex', $('#rev_opex').val());
    $.ajax({
        url: `/predict/${sessionId}`,
        type: 'POST',
        data: fd,
        processData: false,
        contentType: false,
        success: function(res) {
            $('#resRev').fadeIn().removeClass('hidden');
            $('#valRev').text(fmt.format(res.predicted_revenue));
        },
        error: function() { alert('Revenue prediction failed'); }
    });
});

// --- Cost Prediction ---
$('#btnCost').click(function() {
    if (!sessionId) { alert('Upload data first'); return; }
    let fd = new FormData();
    fd.append('diesel', $('#cost_diesel').val());
    fd.append('kwh', $('#cost_kwh').val());
    fd.append('maint', $('#cost_maint').val());
    fd.append('repair', 0);
    fd.append('visits', 0);
    $.ajax({
        url: `/predict_cost/${sessionId}`,
        type: 'POST',
        data: fd,
        processData: false,
        contentType: false,
        success: function(res) {
            $('#resCost').fadeIn().removeClass('hidden');
            $('#valCost').text(fmt.format(res.prediction));
        },
        error: function() { alert('Cost prediction endpoint not available or model not trained'); }
    });
});

// --- Classification ---
$('#btnClf').click(function() {
    if (!sessionId) { alert('Upload data first'); return; }
    let fd = new FormData();
    fd.append('active_tenants', $('#clf_tenants').val());
    fd.append('energy_cost', $('#clf_energy').val());
    fd.append('opex', $('#clf_opex').val());
    $.ajax({
        url: `/predict_class/${sessionId}`,
        type: 'POST',
        data: fd,
        processData: false,
        contentType: false,
        success: function(res) {
            $('#resClf').fadeIn().removeClass('hidden');
            $('#valClf').text(res.prediction);
        },
        error: function() { alert('Classification failed'); }
    });
});

// --- AI Recommendations ---
$('#getRecommendationsBtn').click(function() {
    if (!sessionId) { alert('Please upload data first'); return; }
    $(this).prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Generating...');
    $.ajax({
        url: `/recommend/${sessionId}`,
        type: 'POST',
        success: function(res) {
            let html = res.recommendations;
            if (typeof marked !== 'undefined') html = marked.parse(html);
            $('#recommendations').html(`<div class="prose prose-invert max-w-none text-gray-300">${html}</div>`).show();
            $(this).prop('disabled', false).html('<i class="fas fa-robot"></i> Get Gemini Recommendations');
        }.bind(this),
        error: function() {
            alert('Failed to get recommendations');
            $(this).prop('disabled', false).html('<i class="fas fa-robot"></i> Get Gemini Recommendations');
        }.bind(this)
    });
});

// --- Chat functions ---
function addChatMessage(role, text) {
    let formatted = formatMarkdown(text);
    let alignClass = role === 'user' ? 'justify-end' : 'justify-start';
    let bgClass = role === 'user' ? 'bg-blue-600' : 'bg-gray-800';
    let msgDiv = `<div class="chat-message flex ${alignClass} mb-4">
                    <div class="${bgClass} rounded-2xl px-4 py-2 max-w-[75%]">${formatted}</div>
                  </div>`;
    $('#chatMessages').append(msgDiv);
    $('#chatMessages').scrollTop($('#chatMessages')[0].scrollHeight);
}

function addLoadingSpinner() {
    $('#chatSpinner').remove();
    let spinnerDiv = `<div id="chatSpinner" class="chat-message flex justify-start mb-4">
                        <div class="bg-gray-800 rounded-2xl px-4 py-2"><i class="fas fa-spinner fa-spin"></i> Thinking...</div>
                      </div>`;
    $('#chatMessages').append(spinnerDiv);
    $('#chatMessages').scrollTop($('#chatMessages')[0].scrollHeight);
}

function removeLoadingSpinner() {
    $('#chatSpinner').remove();
}

$('#sendChatBtn').click(function() {
    let msg = $('#chatInput').val().trim();
    if (!msg) return;
    if (!sessionId) { addChatMessage('assistant', 'Please upload data first.'); return; }
    addChatMessage('user', msg);
    $('#chatInput').val('');
    addLoadingSpinner();
    let fd = new FormData();
    fd.append('message', msg);
    $.ajax({
        url: `/chat/${sessionId}`,
        type: 'POST',
        data: fd,
        processData: false,
        contentType: false,
        timeout: 60000,
        success: function(res) {
            removeLoadingSpinner();
            addChatMessage('assistant', res.reply);
        },
        error: function(xhr) {
            removeLoadingSpinner();
            let errMsg = xhr.status === 408 ? 'Request timeout. Try again.' : 'Error: Could not reach AI service.';
            addChatMessage('assistant', errMsg);
        }
    });
});

$('#chatInput').keypress(function(e) { if (e.which === 13) $('#sendChatBtn').click(); });

// --- Tab switching ---
$('.tab-btn').click(function() {
    let tab = $(this).data('tab');
    $('.tab-btn').removeClass('bg-blue-600/30 text-white').addClass('bg-white/5 text-gray-300');
    $(this).addClass('bg-blue-600/30 text-white');
    $('.tab-content').addClass('hidden');
    $(`#tab-${tab}`).removeClass('hidden');
    if (tab === 'track' && storedPlots && !plotsRendered) {
        setTimeout(renderPlots, 50);
    }
});

// Initialize active tab style
$('.tab-btn[data-tab="track"]').addClass('bg-blue-600/30 text-white');

// ========== DYNAMIC FILTERS ==========
// Populate factory and shift dropdowns from the dataset
function populateFilters() {
    if (!sessionId) return;
    $.ajax({
        url: `/filter_options/${sessionId}`,
        type: 'GET',
        success: function(data) {
            let factorySelect = $('#filter_factory');
            factorySelect.empty();
            factorySelect.append('<option value="">All Factories</option>');
            if (data.factories && Array.isArray(data.factories)) {
                data.factories.forEach(f => factorySelect.append(`<option value="${f}">${f}</option>`));
            }
            let shiftSelect = $('#filter_shift');
            shiftSelect.empty();
            shiftSelect.append('<option value="">All Shifts</option>');
            if (data.shifts && Array.isArray(data.shifts)) {
                data.shifts.forEach(s => shiftSelect.append(`<option value="${s}">${s}</option>`));
            }
        },
        error: function() {
            console.warn('Could not load filter options');
        }
    });
}

// Show/hide custom date range inputs based on time range selection
$('#filter_time').change(function() {
    if ($(this).val() === 'CUSTOM') {
        $('#custom_date_range').show();
    } else {
        $('#custom_date_range').hide();
    }
});

// ----- Apply filters (can be called on button click or automatically) -----
function applyFilters() {
    if (!sessionId) return;
    let fd = new FormData();
    fd.append('session_id', sessionId);

    // Time range
    let timeVal = $('#filter_time').val();
    let startDate = null, endDate = null;
    if (timeVal === 'CUSTOM') {
        startDate = $('#start_date').val();
        endDate = $('#end_date').val();
    } else if (timeVal !== 'ALL') {
        let end = new Date();
        let start = new Date();
        if (timeVal === '1D') start.setDate(end.getDate() - 1);
        else if (timeVal === '1W') start.setDate(end.getDate() - 7);
        else if (timeVal === '1M') start.setMonth(end.getMonth() - 1);
        else if (timeVal === '1Y') start.setFullYear(end.getFullYear() - 1);
        startDate = start.toISOString().split('T')[0];
        endDate = end.toISOString().split('T')[0];
    }
    if (startDate && endDate) {
        fd.append('date_range_start', startDate);
        fd.append('date_range_end', endDate);
    }

    // Factory and shift
    let factory = $('#filter_factory').val();
    if (factory) fd.append('factory', factory);
    let shift = $('#filter_shift').val();
    if (shift) fd.append('shift', shift);

    $('#loading').show();
    $.ajax({
        url: '/filter',
        type: 'POST',
        data: fd,
        processData: false,
        contentType: false,
        success: function(res) {
            $('#loading').hide();
            if (res.kpis) displayFilteredKPIs(res.kpis);
            // Update charts
            if (res.line_chart) {
                let fig = JSON.parse(res.line_chart);
                Plotly.newPlot('filter_line_chart', fig.data, fig.layout, {responsive: true});
            }
            if (res.stacked_bar) {
                let fig = JSON.parse(res.stacked_bar);
                Plotly.newPlot('filter_stacked_bar', fig.data, fig.layout, {responsive: true});
            }
            if (res.heatmap) {
                let fig = JSON.parse(res.heatmap);
                Plotly.newPlot('filter_heatmap', fig.data, fig.layout, {responsive: true});
            }
            if (res.pie_energy) {
                let fig = JSON.parse(res.pie_energy);
                Plotly.newPlot('filter_pie', fig.data, fig.layout, {responsive: true});
            }
        },
        error: function(xhr) {
            $('#loading').hide();
            alert('Filter error: ' + (xhr.responseJSON?.detail || 'Unknown error'));
        }
    });
}

// Attach click handler to Apply Filters button
$('#applyFiltersBtn').click(function() {
    applyFilters();
});

// ========== FORECAST IN PREDICT TAB ==========
$('#forecastBtn').click(function() {
    if (!sessionId) { alert('Please upload data first'); return; }
    let days = $('#forecast_days').val();
    let fd = new FormData();
    fd.append('session_id', sessionId);
    fd.append('days', days);
    $('#loading').show();
    $.ajax({
        url: `/forecast/${sessionId}`,
        type: 'POST',
        data: fd,
        processData: false,
        contentType: false,
        success: function(res) {
            $('#loading').hide();
            // Backend returns { future_dates, future_pred, total_next_month }
            // Transform to predictions array
            let predictions = [];
            if (res.future_pred && Array.isArray(res.future_pred)) {
                predictions = res.future_pred.map((value, idx) => ({ day: idx + 1, value: value }));
            } else if (res.predictions) {
                predictions = res.predictions;
            } else {
                alert('Forecast data format not recognized');
                return;
            }
            let x = predictions.map(p => p.day);
            let y = predictions.map(p => p.value);
            let trace = {
                x: x,
                y: y,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Forecasted Cost',
                line: { color: '#8b5cf6', width: 2 },
                marker: { size: 6, color: '#c084fc' }
            };
            let layout = {
                title: `Cost Forecast (${days} Days)`,
                xaxis: { title: 'Days ahead', gridcolor: '#334155' },
                yaxis: { title: 'Cost ($)', gridcolor: '#334155' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#cbd5e1' }
            };
            Plotly.newPlot('forecastChart', [trace], layout, {responsive: true});
        },
        error: function(xhr) {
            $('#loading').hide();
            alert('Forecast error: ' + (xhr.responseJSON?.detail || 'Unknown error'));
        }
    });
});