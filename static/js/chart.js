$(document).ready(function () {
  const student_id = localStorage.getItem("student_id");

  if (!student_id) {
    alert("Student ID not found. Please login first.");
    window.location.href = "/";
    return;
  }

  $.get(`/get_dashboard?student_id=${student_id}`, function (data) {
    const topicData = data.topic_accuracy || [];
    const bloomData = data.bloom_accuracy || [];
    const dktTimeline = data.dkt_timeline || [];
    const dktScore = data.dkt_score;

    // ✅ Display DKT Mastery Score
    if (dktScore !== null && dktScore !== undefined) {
      $('#dktScore').text(`AI-Predicted Mastery Score (DKT): ${(dktScore * 100).toFixed(2)}%`);
    } else {
      $('#dktScore').text("AI-Predicted Mastery Score (DKT): N/A");
    }

    // ✅ Topic-wise Accuracy Bar Chart
    if (topicData.length > 0 && document.getElementById('topicChart')) {
      const topicLabels = topicData.map(item => item.topic);
      const topicValues = topicData.map(item => parseFloat((item.accuracy * 100).toFixed(2)));

      new Chart(document.getElementById('topicChart'), {
        type: 'bar',
        data: {
          labels: topicLabels,
          datasets: [{
            label: '📘 Topic Accuracy (%)',
            data: topicValues,
            backgroundColor: 'rgba(75, 192, 192, 0.6)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: false }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              title: { display: true, text: 'Accuracy (%)' }
            }
          }
        }
      });
    }

    // ✅ Bloom-Level Accuracy Bar Chart
    if (bloomData.length > 0 && document.getElementById('bloomChart')) {
      const bloomLabels = bloomData.map(item => item.bloom);
      const bloomValues = bloomData.map(item => parseFloat((item.accuracy * 100).toFixed(2)));

      new Chart(document.getElementById('bloomChart'), {
        type: 'bar',
        data: {
          labels: bloomLabels,
          datasets: [{
            label: '🎯 Bloom Level Accuracy (%)',
            data: bloomValues,
            backgroundColor: 'rgba(153, 102, 255, 0.6)',
            borderColor: 'rgba(153, 102, 255, 1)',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: false }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              title: { display: true, text: 'Accuracy (%)' }
            }
          }
        }
      });
    }

    // ✅ DKT Line Chart for Score Trend
    if (dktTimeline.length > 0 && document.getElementById('dktLineChart')) {
      const questionLabels = dktTimeline.map((_, idx) => `Q${idx + 1}`);
      const dktValues = dktTimeline.map(score => parseFloat((score * 100).toFixed(2)));

      new Chart(document.getElementById('dktLineChart'), {
        type: 'line',
        data: {
          labels: questionLabels,
          datasets: [{
            label: '📈 DKT Score Trend (%)',
            data: dktValues,
            fill: false,
            borderColor: 'rgba(255, 99, 132, 0.9)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            tension: 0.3,
            pointRadius: 3,
            pointHoverRadius: 6
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              display: true,
              position: 'top'
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              title: { display: true, text: 'Mastery (%)' }
            }
          }
        }
      });
    }

  }).fail(() => {
    alert("⚠️ Could not load dashboard data. Please try again later.");
  });
});
