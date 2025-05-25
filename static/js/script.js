function analyzeAndHighlight(speechText) {
    fetch('/analyze_results', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speech: speechText })
    })
    .then(response => response.json())
    .then(data => {
      const resultContainer = document.getElementById('recitedAnswer');
      resultContainer.innerHTML = ''; // Clear any previous content
  
      data.forEach(item => {
        const span = document.createElement('span');
        span.textContent = item.word + ' ';
        span.style.color = item.closed ? 'red' : 'green';
        resultContainer.appendChild(span);
      });
    })
    .catch(error => console.error('Error:', error));
  }
  