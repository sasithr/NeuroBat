/* =========================================================
   NeuroBat Global Navigation
   ========================================================= */

(function () {

  const mount =
    document.getElementById("neurobatNavbar");

  if (!mount) {
    return;
  }


  const currentPath =
    window.location.pathname;


  function activeClass(paths) {

    return paths.includes(currentPath)
      ? " active"
      : "";

  }


  function escapeHtml(value) {

    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  }


  function initials(name) {

    const parts =
      String(name || "")
        .trim()
        .split(/\s+/)
        .filter(Boolean);


    if (!parts.length) {
      return "NB";
    }


    if (parts.length === 1) {

      return parts[0]
        .slice(0, 2)
        .toUpperCase();

    }


    return (
      parts[0][0] +
      parts[parts.length - 1][0]
    ).toUpperCase();

  }


  function firstName(name) {

    const parts =
      String(name || "")
        .trim()
        .split(/\s+/)
        .filter(Boolean);


    return (
      parts[0] ||
      "Account"
    );

  }


  function renderBaseNavigation() {

    mount.innerHTML = `
      <nav
        class="nb-navbar"
        aria-label="Main navigation"
      >

        <a
          class="nb-logo"
          href="/"
          aria-label="NeuroBat home"
        >
          Neuro<span>Bat</span>
        </a>


        <button
          class="nb-mobile-toggle"
          id="nbMobileToggle"
          type="button"
          aria-label="Open navigation menu"
          aria-expanded="false"
        >
          ☰
        </button>


        <div
          class="nb-nav-center"
          id="nbNavCenter"
        >

          <a
            class="nb-nav-link${activeClass([
              "/features",
              "/feature.html",
              "/features.html"
            ])}"
            href="/features"
          >
            Features
          </a>


          <a
            class="nb-nav-link${activeClass([
              "/analytics",
              "/analytics.html"
            ])}"
            href="/analytics"
          >
            Analytics
          </a>


          <a
            class="nb-nav-link${activeClass([
              "/history",
              "/history.html"
            ])}"
            href="/history"
          >
            History
          </a>


          <a
            class="nb-nav-link${activeClass([
              "/pricing",
              "/pricing.html"
            ])}"
            href="/pricing"
          >
            Pricing
          </a>


          <a
            class="nb-nav-link${activeClass([
              "/contact",
              "/contact.html"
            ])}"
            href="/contact"
          >
            Contact
          </a>

        </div>


        <div
          class="nb-nav-actions"
          id="nbNavActions"
        >

          <a
            class="nb-btn nb-btn-ghost nb-auth-login"
            href="/login"
          >
            Login
          </a>


          <a
            class="nb-btn"
            href="/register"
          >
            Register
          </a>

        </div>

      </nav>
    `;


    const mobileToggle =
      document.getElementById(
        "nbMobileToggle"
      );


    const navCenter =
      document.getElementById(
        "nbNavCenter"
      );


    if (
      mobileToggle &&
      navCenter
    ) {

      mobileToggle.addEventListener(
        "click",
        function () {

          const isOpen =
            navCenter.classList.toggle(
              "open"
            );


          mobileToggle.setAttribute(
            "aria-expanded",
            String(isOpen)
          );

        }
      );

    }

  }


  function renderSignedOut() {

    const actions =
      document.getElementById(
        "nbNavActions"
      );


    if (!actions) {
      return;
    }


    actions.innerHTML = `
      <a
        class="nb-btn nb-btn-ghost nb-auth-login"
        href="/login"
      >
        Login
      </a>


      <a
        class="nb-btn"
        href="/register"
      >
        Register
      </a>
    `;

  }


  function renderSignedIn(user) {

    const actions =
      document.getElementById(
        "nbNavActions"
      );


    if (!actions) {
      return;
    }


    const name =
      user.full_name ||
      "NeuroBat User";


    const email =
      user.email ||
      "";


    actions.innerHTML = `
      <a
        class="nb-btn nb-analyze-button"
        href="/analyze-page"
      >
        Analyze
      </a>


      <div class="nb-account-wrap">

        <button
          class="nb-account-button"
          id="nbAccountButton"
          type="button"
          aria-expanded="false"
          aria-controls="nbAccountMenu"
        >

          <span class="nb-avatar">
            ${escapeHtml(
              initials(name)
            )}
          </span>


          <span class="nb-account-label">
            ${escapeHtml(
              firstName(name)
            )}
          </span>


          <span class="nb-chevron">
            ▼
          </span>

        </button>


        <div
          class="nb-account-menu"
          id="nbAccountMenu"
        >

          <div class="nb-account-kicker">
            Signed in
          </div>


          <div class="nb-account-name">
            ${escapeHtml(name)}
          </div>


          <div class="nb-account-email">
            ${escapeHtml(email)}
          </div>


          <div class="nb-menu-divider"></div>


          <div class="nb-menu-links">

            <a
              class="nb-menu-link"
              href="/player-info"
            >
              Profile
            </a>


            <a
              class="nb-menu-link"
              href="/analyze-page"
            >
              Analyze Video
            </a>


            <a
              class="nb-menu-link"
              href="/history"
            >
              Analysis History
            </a>


            <a
              class="nb-menu-link nb-menu-link-danger"
              href="/logout"
            >
              Logout
            </a>

          </div>

        </div>

      </div>
    `;


    const accountButton =
      document.getElementById(
        "nbAccountButton"
      );


    const accountMenu =
      document.getElementById(
        "nbAccountMenu"
      );


    if (
      accountButton &&
      accountMenu
    ) {

      accountButton.addEventListener(
        "click",
        function (event) {

          event.stopPropagation();


          const isOpen =
            accountMenu.classList.toggle(
              "open"
            );


          accountButton.setAttribute(
            "aria-expanded",
            String(isOpen)
          );

        }
      );


      document.addEventListener(
        "click",
        function (event) {

          if (
            !accountMenu.contains(
              event.target
            ) &&
            !accountButton.contains(
              event.target
            )
          ) {

            accountMenu.classList.remove(
              "open"
            );


            accountButton.setAttribute(
              "aria-expanded",
              "false"
            );

          }

        }
      );


      document.addEventListener(
        "keydown",
        function (event) {

          if (
            event.key ===
            "Escape"
          ) {

            accountMenu.classList.remove(
              "open"
            );


            accountButton.setAttribute(
              "aria-expanded",
              "false"
            );

          }

        }
      );

    }


    const getStartedBtn =
      document.getElementById(
        "getStartedBtn"
      );


    if (getStartedBtn) {

      getStartedBtn.href =
        "/player-info";


      getStartedBtn.textContent =
        "Continue";

    }

  }


  async function loadUser() {

    try {

      const response =
        await fetch(
          "/api/me",
          {
            headers: {
              "Accept":
                "application/json"
            }
          }
        );


      if (!response.ok) {

        renderSignedOut();

        return;

      }


      const data =
        await response.json();


      if (
        !data ||
        !data.user
      ) {

        renderSignedOut();

        return;

      }


      renderSignedIn(
        data.user
      );

    }


    catch (error) {

      console.error(
        "NeuroBat navigation user lookup failed:",
        error
      );


      renderSignedOut();

    }

  }


  renderBaseNavigation();

  loadUser();

})();