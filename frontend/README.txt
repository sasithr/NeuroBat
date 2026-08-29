NEUROBAT LANDING PAGE — FLASK / VS CODE PACKAGE

1. Copy landingpage.html, styles.css, script.js and the neurobat folder into:
   backend/frontend/

2. Keep all four items together. The page uses relative media paths such as:
   neurobat/biomechanics.png

3. Your existing Flask routes are already preserved:
   /register
   /login
   /features
   /analyze-page
   /history
   /progress
   /api/me

4. The page does not require Bootstrap, jQuery, Slick or AOS. Their useful
   behavior is implemented with modern CSS, native JavaScript and
   IntersectionObserver.

5. Test from the Flask server rather than opening the HTML with file://,
   because the signed-in CTA check calls /api/me.
