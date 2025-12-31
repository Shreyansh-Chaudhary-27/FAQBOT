document.addEventListener('DOMContentLoaded', function () {
    const toggleBtn = document.querySelector('.toggle-sidebar-btn');
    const sidebar = document.querySelector('.sidebar');
    if (localStorage.getItem('sidebar-collapsed') === 'true') {
        sidebar?.classList.add('collapsed');
    }
    toggleBtn?.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        localStorage.setItem('sidebar-collapsed', sidebar.classList.contains('collapsed'));
    });
});
