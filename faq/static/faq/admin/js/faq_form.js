document.addEventListener('DOMContentLoaded', function () {
            const toggleBtn = document.querySelector('.toggle-sidebar-btn');
            const sidebar = document.querySelector('.sidebar');

            function isMobile() {
                return window.innerWidth <= 768;
            }

            // Desktop state restore ONLY
            if (!isMobile()) {
                const isCollapsed = localStorage.getItem('sidebar-collapsed') === 'true';
                if (isCollapsed) {
                    sidebar.classList.add('collapsed');
                }
            }

            toggleBtn.addEventListener('click', function () {

                if (isMobile()) {
                    // Mobile toggle
                    sidebar.classList.toggle('mobile-open');
                } else {
                    // Desktop toggle
                    sidebar.classList.toggle('collapsed');
                    localStorage.setItem(
                        'sidebar-collapsed',
                        sidebar.classList.contains('collapsed')
                    );
                }
            });

            // On resize: clean states
            window.addEventListener('resize', function () {
                if (isMobile()) {
                    sidebar.classList.remove('collapsed');
                } else {
                    sidebar.classList.remove('mobile-open');
                }
            });
        });