(function () {
  "use strict";

  /* =========================================================
     Reveal animations
     ========================================================= */

  var revealElements =
    document.querySelectorAll("[data-reveal]");

  if (
    window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
    !("IntersectionObserver" in window)
  ) {
    revealElements.forEach(function (element) {
      element.classList.add("is-visible");
    });
  } else {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.14 }
    );

    revealElements.forEach(function (element) {
      observer.observe(element);
    });
  }


  /* =========================================================
     Feature gallery
     ========================================================= */

  var galleryData = [
    {
      label: "Pose tracking",
      title: "See movement frame by frame.",
      copy:
        "Body landmarks convert a batting clip into movement evidence that can be reviewed, compared and explained.",
      facts: [
        "33 body landmarks",
        "Frame-level tracking",
        "Visual evidence"
      ]
    },
    {
      label: "Biomechanics",
      title: "Turn motion into meaningful metrics.",
      copy:
        "Review head position, shoulder alignment, knee flexion and movement in the context of the full stroke.",
      facts: [
        "Joint angles",
        "Alignment metrics",
        "Movement indicators"
      ]
    },
    {
      label: "Shot library",
      title: "Build a clearer picture of technique.",
      copy:
        "Keep visual examples and session results together so players can discuss the movement—not just the outcome.",
      facts: [
        "Session history",
        "Shot context",
        "Progress view"
      ]
    }
  ];

  var galleryButtons =
    Array.from(
      document.querySelectorAll(
        ".gallery-tabs button"
      )
    );

  var galleryImages =
    Array.from(
      document.querySelectorAll(
        ".gallery-media > img"
      )
    );

  var galleryIndex =
    document.querySelector(
      ".gallery-index"
    );

  var galleryLabel =
    document.querySelector(
      ".gallery-copy > span"
    );

  var galleryTitle =
    document.querySelector(
      ".gallery-copy h3"
    );

  var galleryCopy =
    document.querySelector(
      ".gallery-copy > p"
    );

  var galleryFacts =
    document.querySelector(
      ".gallery-copy ul"
    );


  function showGallery(index) {
    var item = galleryData[index];

    if (!item) {
      return;
    }

    galleryButtons.forEach(
      function (button, buttonIndex) {
        var selected =
          buttonIndex === index;

        button.classList.toggle(
          "is-active",
          selected
        );

        button.setAttribute(
          "aria-selected",
          String(selected)
        );
      }
    );

    galleryImages.forEach(
      function (image, imageIndex) {
        image.classList.toggle(
          "is-active",
          imageIndex === index
        );
      }
    );

    if (galleryIndex) {
      galleryIndex.textContent =
        "0" + (index + 1) + " / 03";
    }

    if (galleryLabel) {
      galleryLabel.textContent =
        item.label;
    }

    if (galleryTitle) {
      galleryTitle.textContent =
        item.title;
    }

    if (galleryCopy) {
      galleryCopy.textContent =
        item.copy;
    }

    if (galleryFacts) {
      galleryFacts.innerHTML =
        item.facts
          .map(function (fact) {
            return (
              "<li><i></i>" +
              fact +
              "</li>"
            );
          })
          .join("");
    }
  }


  galleryButtons.forEach(
    function (button, index) {
      button.addEventListener(
        "click",
        function () {
          showGallery(index);
        }
      );
    }
  );


  /* =========================================================
     FAQ
     ========================================================= */

  document
    .querySelectorAll(
      ".faq-list article"
    )
    .forEach(
      function (article) {
        var button =
          article.querySelector(
            "button"
          );

        var answer =
          article.querySelector(
            ".faq-answer"
          );

        if (!button || !answer) {
          return;
        }

        button.addEventListener(
          "click",
          function () {
            var open =
              !article.classList.contains(
                "is-open"
              );

            document
              .querySelectorAll(
                ".faq-list article"
              )
              .forEach(
                function (item) {
                  item.classList.remove(
                    "is-open"
                  );

                  var itemButton =
                    item.querySelector(
                      "button"
                    );

                  var itemAnswer =
                    item.querySelector(
                      ".faq-answer"
                    );

                  if (itemButton) {
                    itemButton.setAttribute(
                      "aria-expanded",
                      "false"
                    );
                  }

                  if (itemAnswer) {
                    itemAnswer.setAttribute(
                      "aria-hidden",
                      "true"
                    );
                  }
                }
              );

            article.classList.toggle(
              "is-open",
              open
            );

            button.setAttribute(
              "aria-expanded",
              String(open)
            );

            answer.setAttribute(
              "aria-hidden",
              String(!open)
            );
          }
        );
      }
    );


  /* =========================================================
     Landing CTA authentication state
     ========================================================= */

  fetch(
    "/api/me",
    {
      credentials: "same-origin",
      cache: "no-store",
      headers: {
        Accept:
          "application/json"
      }
    }
  )
    .then(function (response) {
      if (!response.ok) {
        throw new Error(
          "Signed out"
        );
      }

      return response.json();
    })
    .then(function (data) {
      if (!data || !data.user) {
        throw new Error(
          "Signed out"
        );
      }

      document
        .querySelectorAll(
          "[data-auth-cta]"
        )
        .forEach(
          function (link) {
            link.setAttribute(
              "href",
              "/analyze-page"
            );

            if (
              link.dataset.authCta ===
              "hero"
            ) {
              link.innerHTML =
                'Analyze your batting <span aria-hidden="true">→</span>';
            } else {
              link.innerHTML =
                'Start new analysis <span aria-hidden="true">→</span>';
            }
          }
        );
    })
    .catch(function () {
      /* Logged-out CTA state is already correct. */
    });

})();