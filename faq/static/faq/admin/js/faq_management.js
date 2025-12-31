document.addEventListener('DOMContentLoaded', function () {

    // Sidebar Logic
    const toggleBtn = document.querySelector('.toggle-sidebar-btn');
    const sidebar = document.querySelector('.sidebar');
    const isCollapsed = localStorage.getItem('sidebar-collapsed') === 'true';
    if (isCollapsed) sidebar?.classList.add('collapsed');

    toggleBtn?.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
    });

    // Delete FAQ Confirmation
    window.deleteFAQ = function (id) {
        if (confirm('Are you sure you want to delete this FAQ entry?')) {
            const csrfToken = document.querySelector('input[name="csrfmiddlewaretoken"]')?.value;
            const url = `/admin-dashboard/faq/${id}/delete/`;

            fetch(url, {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken,
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
                .then(response => {
                    if (response.ok) {
                        window.location.reload();
                    } else {
                        alert('Failed to delete FAQ entry.');
                    }
                })
                .catch(err => alert('Failed to delete item.'));
        }
    };

    // --- Frontend Filter Logic ---
    const filterBtn = document.getElementById('filter-btn');
    const searchInput = document.getElementById('search-input');
    const keywordInput = document.getElementById('keyword-input');
    const tableBody = document.getElementById('faq-table-body');

    function applyFilters() {
        if (!tableBody) return;
        const rows = tableBody.querySelectorAll('tr');
        const searchTerm = searchInput?.value.toLowerCase().trim() || '';
        const keywordTerm = keywordInput?.value.toLowerCase().trim() || '';

        rows.forEach(row => {
            // Check if it's the "No FAQ entries found" empty row
            if (row.cells.length === 1 && row.cells[0].colSpan > 1) return;

            // Get text from specific columns
            // Column 1: Question, Column 2: Answer, Column 3: Keywords
            const questionText = row.querySelector('.faq-question')?.textContent.toLowerCase() || '';
            const answerText = row.querySelector('.faq-answer')?.textContent.toLowerCase() || '';

            // Collect keywords from badges
            const badges = row.querySelectorAll('.keyword-badge');
            let keywordsText = '';
            badges.forEach(b => keywordsText += b.textContent.toLowerCase() + ' ');

            const matchesSearch = !searchTerm || questionText.includes(searchTerm) || answerText.includes(searchTerm);
            const matchesKeyword = !keywordTerm || keywordsText.includes(keywordTerm);

            if (matchesSearch && matchesKeyword) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }

    // Filter on button click
    filterBtn?.addEventListener('click', applyFilters);

    // Optional: Filter on Enter key in inputs
    searchInput?.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') applyFilters();
    });
    keywordInput?.addEventListener('keyup', (e) => {
        if (e.key === 'Enter') applyFilters();
    });


    // --- CSV Export Logic ---
    const csvBtn = document.getElementById('csv-btn');

    csvBtn?.addEventListener('click', function (e) {
        e.preventDefault();
        exportTableToCSV('faq_data_export.csv');
    });

    function exportTableToCSV(filename) {
        if (!tableBody) return;
        const rows = Array.from(tableBody.querySelectorAll('tr'));
        let csv = [];

        // Add Header
        csv.push(['ID', 'Question', 'Answer', 'Keywords'].map(escapeCSVVal).join(','));

        rows.forEach(row => {
            if (row.style.display === 'none') return; // Skip hidden rows
            if (row.cells.length === 1 && row.cells[0].colSpan > 1) return; // Skip empty message

            const cols = row.querySelectorAll('td');
            if (cols.length < 4) return;

            const id = cols[0].innerText.trim();
            const question = cols[1].innerText.trim();
            const answer = cols[2].innerText.trim();
            // Keywords are in badges
            const badges = cols[3].querySelectorAll('.keyword-badge');
            let keywords = [];
            badges.forEach(b => keywords.push(b.innerText.trim()));

            csv.push([id, question, answer, keywords.join('; ')].map(escapeCSVVal).join(','));
        });

        downloadCSV(csv.join('\n'), filename);
    }

    function escapeCSVVal(text) {
        if (!text) return '""';
        return '"' + text.toString().replace(/"/g, '""') + '"';
    }

    function downloadCSV(csv, filename) {
        const csvFile = new Blob([csv], { type: 'text/csv' });
        const downloadLink = document.createElement('a');
        downloadLink.download = filename;
        downloadLink.href = window.URL.createObjectURL(csvFile);
        downloadLink.style.display = 'none';
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    }
});
