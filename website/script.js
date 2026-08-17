const patterns = {
  input: [
    "dim", "dim", "hot", "dim", "hot", "dim", "dim",
    "dim", "hot", "dim", "dim", "dim", "hot", "dim",
    "hot", "dim", "acid", "dim", "acid", "dim", "hot",
    "dim", "dim", "dim", "hot", "dim", "dim", "dim",
    "", "", "", "", "", "", "",
    "", "", "", "", "", "", "",
    "", "", "", "", "", "", ""
  ],
  output: [
    "dim", "dim", "hot", "dim", "hot", "dim", "dim",
    "dim", "hot", "dim", "dim", "dim", "hot", "dim",
    "hot", "dim", "acid", "dim", "acid", "dim", "hot",
    "dim", "dim", "dim", "hot", "dim", "dim", "dim",
    "hot", "dim", "acid", "dim", "acid", "dim", "hot",
    "dim", "hot", "dim", "dim", "dim", "hot", "dim",
    "dim", "dim", "hot", "dim", "hot", "dim", "dim"
  ]
};

function fillGrid(selector, pattern) {
  const grid = document.querySelector(selector);
  pattern.forEach((className) => {
    const cell = document.createElement("span");
    if (className) cell.className = className;
    grid.appendChild(cell);
  });
}

fillGrid(".input-grid", patterns.input);
fillGrid(".output-grid", patterns.output);

const tabs = document.querySelectorAll(".compare-tab");
const panels = document.querySelectorAll(".method-panel");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });

    panels.forEach((panel) => {
      const selected = panel.id === `panel-${tab.dataset.target}`;
      panel.hidden = !selected;
      panel.classList.toggle("active", selected);
    });
  });
});

const charts = document.querySelectorAll(".chart-card");
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add("is-visible");
    });
  },
  { threshold: 0.35 }
);

charts.forEach((chart) => observer.observe(chart));
