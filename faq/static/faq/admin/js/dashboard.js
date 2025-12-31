document.addEventListener('DOMContentLoaded', function () {

    // Sidebar Logic
    const toggleBtn = document.querySelector('.toggle-sidebar-btn');
    const sidebar = document.querySelector('.sidebar');

    // Check local storage for sidebar state
    const isCollapsed = localStorage.getItem('sidebar-collapsed') === 'true';
    if (isCollapsed) {
        sidebar?.classList.add('collapsed');
    }

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', function () {
            sidebar.classList.toggle('collapsed');
            localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
        });
    }

    // Chart.js Logic
    if (typeof Chart !== 'undefined') {
        const activityCtx = document.getElementById('activityChart');
        const statusCtx = document.getElementById('statusChart');

        // Data from DOM attributes (passed from template)
        const recentRequests = parseInt(document.getElementById('chart-data').dataset.recentRequests || 0);
        const totalRequests = parseInt(document.getElementById('chart-data').dataset.totalRequests || 0);
        const totalResponses = parseInt(document.getElementById('chart-data').dataset.totalResponses || 0);
        const pendingRequests = parseInt(document.getElementById('chart-data').dataset.pendingRequests || 0);
        const answeredRequests = Math.max(0, totalRequests - pendingRequests);

        // Bar Chart - Recent Activity
        if (activityCtx) {
            new Chart(activityCtx, {
                type: 'bar',
                data: {
                    labels: ['Requests', 'Responses'],
                    datasets: [{
                        label: 'Last 7 Days',
                        data: [recentRequests, parseInt(document.getElementById('chart-data').dataset.recentResponses || 0)],
                        backgroundColor: '#4F46E5',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { beginAtZero: true, grid: { borderDash: [2, 2] } },
                        x: { grid: { display: false } }
                    }
                }
            });
        }

        // Doughnut Chart - Request Status
        if (statusCtx) {
            new Chart(statusCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Answered', 'Pending'],
                    datasets: [{
                        data: [totalResponses, pendingRequests],
                        backgroundColor: ['#10B981', '#EF4444'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '75%',
                    plugins: {
                        legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8 } }
                    }
                }
            });
        }
    }
});
