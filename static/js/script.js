$(document).ready(function () {
  const page = window.location.pathname;

  // ===========================
  // LOGIN PAGE LOGIC
  // ===========================
  if (page === "/" || page.includes("login")) {
    $('#loginBtn').on('click', function () {
      const name = $('#studentName').val().trim();
      const topic = $('#topic').val();

      if (!name || !topic) {
    alert("Please enter your name and select a topic.");
    return;
  }

  $.ajax({
    url: '/login',
    method: 'POST',
    contentType: 'application/json',
    data: JSON.stringify({ name }),
    success: function (res) {
      localStorage.setItem("student_id", res.student_id);
      localStorage.setItem("topic", topic);
      window.location.href = "/quiz";
    },
    error: function () {
      alert("Login failed. Please try again.");
    }
  });
});
  }

  // ===========================
  // QUIZ PAGE LOGIC
  // ===========================
  if (page.includes("/quiz")) {
  const student_id = localStorage.getItem("student_id");
  const topic = localStorage.getItem("topic");

  if (!student_id || !topic) {
    alert("Session expired. Please login again.");
    window.location.href = "/";
    return;
  }

  // Fetch quiz questions
  $.get(`/get_quiz?topic=${topic}`, function (questions) {
    if (!questions || questions.length === 0) {
      $('#quizForm').html("<p>No questions available for this topic.</p>");
      return;
    }

    // Render each question
    questions.forEach((q, index) => {
      $('#quizForm').append(`
        <div class="mb-3 question-block">
          <label class="form-label">${index + 1}. ${q.question}</label>
          <input type="text" class="form-control" data-qid="${q.id}" placeholder="Your answer" required>
          <div class="correct-answer text-success mt-1" style="display:none;"></div>
          <span class="badge bg-info mt-1">Bloom Level: ${q.bloom_level}</span>
        </div>
      `);
    });
  });

  // Handle submission
  $('#submitQuizBtn').on('click', async function () {
    const answers = [];
    let incomplete = false;

    $('input[data-qid]').each(function () {
      const question_id = $(this).data('qid');
      const user_answer = $(this).val().trim();

      if (!user_answer) {
        alert("Please answer all questions before submitting.");
        incomplete = true;
        return false;  // break loop
      }

      answers.push({
        student_id: parseInt(student_id),
        question_id: parseInt(question_id),
        user_answer
      });
    });

    if (incomplete || answers.length === 0) return;

    // Submit each answer and show feedback
    const requests = answers.map((ans, i) =>
      $.ajax({
        url: '/submit_answer',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(ans)
      }).then((response) => {
        if (!response.correct) {
          const block = $('.question-block').eq(i);
          const answerDiv = block.find('.correct-answer');
          answerDiv.html(`<strong>✅ Correct Answer:</strong> ${response.expected_answer}`);
          answerDiv.show();
        }
      })
    );

    // Wait for all answers, then redirect
    Promise.all(requests)
      .then(() => {
        setTimeout(() => {
          window.location.href = "/dashboard";
        }, 2500); // Wait 2.5 seconds to show correct answers
      })
      .catch(() => {
        alert("❌ Error submitting answers. Please try again.");
      });
  });
}


    // ===========================
  // DASHBOARD PAGE LOGIC
  // ===========================
// DASHBOARD PAGE LOGIC
// ===========================
if (page.includes("/dashboard")) {
  const student_id = localStorage.getItem("student_id");
if (!student_id) {
  alert("Session expired. Please login again.");
  window.location.href = "/";
  return;
}

function loadDashboard(student_id) {
  $.get(`/get_dashboard?student_id=${student_id}`, function (data) {
    if (!data) {
      alert("No dashboard data available.");
      return;
    }

    const topicData = data.topic_accuracy || [];
    if (topicData.length > 0) {
      new Chart(document.getElementById('topicChart'), {
        type: 'bar',
        data: {
          labels: topicData.map(d => d.topic),
          datasets: [{
            label: 'Topic Accuracy (%)',
            data: topicData.map(d => d.accuracy),
            backgroundColor: 'rgba(75, 192, 192, 0.6)'
          }]
        },
        options: {
          scales: {
            y: {
              beginAtZero: true,
              suggestedMin: 0,
              suggestedMax: 100
            }
          },
          plugins: {
            legend: { display: true },
            tooltip: { enabled: true }
          }
        }
      });
    }

    const bloomLabels = data.bloom_accuracy.map(item => item.bloom);
    const bloomValues = data.bloom_accuracy.map(item => parseFloat(item.accuracy.toFixed(2)));

    const bloomCtx = document.getElementById("bloomChart").getContext("2d");

    new Chart(bloomCtx, {
      type: 'bar',
      data: {
        labels: bloomLabels,
        datasets: [{
          label: 'Bloom Accuracy (%)',
          data: bloomValues,
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 100
          }
        }
      }
    });



    if (data.dkt_score !== null && data.dkt_score !== undefined) {
      $('#dktScore').text(`AI-Predicted Mastery Score (DKT): ${(data.dkt_score * 100).toFixed(2)}%`);
    } else {
      $('#dktScore').text("AI-Predicted Mastery Score (DKT): N/A");
    }
    const ctx = document.getElementById("dktLineChart").getContext("2d");

new Chart(ctx, {
  type: "line",
  data: {
    labels: data.timestamps, // e.g., ["Jul 29, 21:35", "Jul 29, 21:37"]
    datasets: [{
      label: "Mastery Over Time",
      data: data.dkt_timeline.map(score => parseFloat((score * 100).toFixed(2))), // convert to %
      fill: false,
      borderColor: "rgba(54, 162, 235, 1)",
      backgroundColor: "rgba(54, 162, 235, 0.2)",
      tension: 0.2
    }]
  },
  options: {
    responsive: true,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: "Mastery (%)"
        }
      },
      x: {
        title: {
          display: true,
          text: "Time"
        }
      }
    }
  }
});
  });
}

loadDashboard(student_id);
}
});
