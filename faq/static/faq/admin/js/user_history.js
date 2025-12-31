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
});

window.deleteUser = function (userId, deleteUrl) {
    if (!confirm('Are you sure you want to delete this user? This action cannot be undone.')) {
        return false;
    }

    // Use fetch to hit the Django admin delete URL with a POST request
    // Note: Django admin delete view usually requires a confirmation POST.
    // If we can't do it cleanly via AJAX because of the admin confirmation page,
    // we might have to compromise. 
    // However, for typical DeleteView, a POST often bypasses confirmation screen if configured.
    // Standard Django admin delete view usually renders a confirmation page on GET
    // and deletes on POST. 

    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

    fetch(deleteUrl, {
        method: 'POST',
        headers: {
            'X-CSRFToken': csrfToken,
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: 'post=yes' // Often used to confirm
    })
        .then(response => {
            // If it managed to delete, it usually redirects to the changelist.
            // We can just reload the current page.
            if (response.ok || response.redirected) {
                window.location.reload();
            } else {
                alert('Failed to delete user.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred.');
        });

    return false;
};
