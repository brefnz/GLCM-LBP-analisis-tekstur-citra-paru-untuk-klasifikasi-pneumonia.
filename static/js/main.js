/**
 * LungClassify – main.js
 * Handles: sidebar toggle, model status badge, utility functions
 */

(function () {
  'use strict';

  /* ── Sidebar toggle ───────────────────────────────────────── */
  const sidebar     = document.getElementById('appSidebar');
  const content     = document.getElementById('appContent');
  const toggleBtn   = document.getElementById('sidebarToggle');

  // Restore preference
  const sidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
  if (sidebarCollapsed) {
    sidebar && sidebar.classList.add('collapsed');
    content && content.classList.add('expanded');
  }

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const isCollapsed = sidebar.classList.toggle('collapsed');
      content.classList.toggle('expanded', isCollapsed);
      localStorage.setItem('sidebarCollapsed', isCollapsed);
    });
  }

  /* ── Sidebar model status pill ────────────────────────────── */
  const statusDot   = document.getElementById('sidebarStatusDot');
  const statusLabel = document.getElementById('sidebarStatusLabel');

  function updateSidebarStatus() {
    fetch('/api/model-info')
      .then(r => r.json())
      .then(data => {
        if (!statusDot || !statusLabel) return;
        if (data.trained) {
          statusDot.className  = 'status-dot trained';
          const acc = data.metrics
            ? (data.metrics.accuracy * 100).toFixed(1) + '%'
            : '';
          statusLabel.textContent = acc ? `Model siap — ${acc}` : 'Model siap';
        } else {
          statusDot.className  = 'status-dot not-trained';
          statusLabel.textContent = 'Belum dilatih';
        }
      })
      .catch(() => {
        if (statusLabel) statusLabel.textContent = 'Tidak terhubung';
      });
  }

  // Expose so train.html can call it after training finishes
  window.updateSidebarStatus = updateSidebarStatus;

  // Poll every 10s (lightweight — just checks file existence)
  updateSidebarStatus();
  setInterval(updateSidebarStatus, 10000);

  /* ── Auto-dismiss flash messages ─────────────────────────── */
  document.querySelectorAll('.flash-msg').forEach(el => {
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transition = 'opacity 0.4s';
      setTimeout(() => el.remove(), 400);
    }, 4000);
  });

})();
