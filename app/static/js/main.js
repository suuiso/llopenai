document.addEventListener('DOMContentLoaded', function () {
  const strategy = document.getElementById('strategy');
  const paramsParagraphs = document.getElementById('params-paragraphs');
  const paramsCharacters = document.getElementById('params-characters');

  function toggleParams() {
    if (!strategy) return;
    const value = strategy.value;
    if (value === 'characters') {
      paramsParagraphs?.classList.add('d-none');
      paramsCharacters?.classList.remove('d-none');
    } else {
      paramsParagraphs?.classList.remove('d-none');
      paramsCharacters?.classList.add('d-none');
    }
  }

  strategy?.addEventListener('change', toggleParams);
  toggleParams();
});

