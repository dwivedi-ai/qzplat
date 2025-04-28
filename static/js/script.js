// stereotype_quiz_app/static/js/script.js

// Wait for the HTML document to be fully loaded before running the script
document.addEventListener('DOMContentLoaded', function() {

    // --- Toggle Subset Visibility ---
    const toggleButtons = document.querySelectorAll('.toggle-subsets');

    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);

            if (targetElement) {
                if (targetElement.style.display === 'none' || targetElement.style.display === '') {
                    targetElement.style.display = 'block'; // Show the subsets
                    this.textContent = 'Hide Details'; // Change button text
                } else {
                    targetElement.style.display = 'none'; // Hide the subsets
                    this.textContent = 'Show Details'; // Change button text back
                }
            } else {
                console.error('Could not find subset target element with ID:', targetId);
            }
        });
    });

    // --- Show/Hide Offensiveness Rating Based on Annotation ---
    const annotationFieldsets = document.querySelectorAll('.annotation-options');

    annotationFieldsets.forEach(fieldset => {
        // Listen for changes within the fieldset (radio button selection)
        fieldset.addEventListener('change', function(event) {
            // Check if the changed element is a radio button and is checked
            if (event.target.type === 'radio' && event.target.checked) {
                const selectedValue = event.target.value;
                const questionIndex = this.getAttribute('data-question-index'); // Get index from fieldset
                const ratingContainer = document.getElementById(`rating_container_${questionIndex}`);
                const ratingRadios = ratingContainer ? ratingContainer.querySelectorAll('input[type="radio"]') : [];

                if (ratingContainer) {
                    if (selectedValue === 'Stereotype') {
                        ratingContainer.style.display = 'block'; // Show the rating section
                        // Make the rating radios required *only when visible*
                        ratingRadios.forEach(radio => radio.required = true);
                    } else {
                        ratingContainer.style.display = 'none'; // Hide the rating section
                        // Unset required and uncheck any selected rating if hiding
                        ratingRadios.forEach(radio => {
                            radio.required = false;
                            radio.checked = false; // Optional: clear selection when hiding
                        });
                    }
                } else {
                    console.error('Could not find rating container for index:', questionIndex);
                }
            }
        });

        // Initial check in case a value is pre-selected (e.g., browser back button)
        // This might not be strictly necessary but can handle edge cases
        const checkedRadio = fieldset.querySelector('input[type="radio"]:checked');
        if (checkedRadio) {
             const questionIndex = fieldset.getAttribute('data-question-index');
             const ratingContainer = document.getElementById(`rating_container_${questionIndex}`);
             const ratingRadios = ratingContainer ? ratingContainer.querySelectorAll('input[type="radio"]') : [];

             if (ratingContainer) {
                 if (checkedRadio.value === 'Stereotype') {
                     ratingContainer.style.display = 'block';
                     ratingRadios.forEach(radio => radio.required = true);
                 } else {
                     ratingContainer.style.display = 'none';
                     ratingRadios.forEach(radio => radio.required = false);
                 }
             }
        }
    });

    // --- Optional: Client-side Form Validation on Submit ---
    const quizForm = document.getElementById('quiz-form');
    if (quizForm) {
        quizForm.addEventListener('submit', function(event) {
            let firstErrorElement = null; // To focus on the first error

            // Re-check all annotation options (HTML required should handle this, but good failsafe)
            annotationFieldsets.forEach(fieldset => {
                const questionIndex = fieldset.getAttribute('data-question-index');
                const radios = fieldset.querySelectorAll('input[type="radio"]');
                const isSelected = Array.from(radios).some(radio => radio.checked);
                if (!isSelected) {
                    console.warn(`Annotation missing for question index ${questionIndex}`);
                    if (!firstErrorElement) firstErrorElement = fieldset;
                    // Optionally add visual error indication here
                }
            });

            // Check required offensiveness ratings *only* for visible sections
            const visibleRatingContainers = document.querySelectorAll('.offensiveness-rating-container[style*="display: block"]');
            visibleRatingContainers.forEach(container => {
                const radios = container.querySelectorAll('input[type="radio"]');
                const isSelected = Array.from(radios).some(radio => radio.checked);
                if (!isSelected) {
                     console.warn(`Offensiveness rating missing for visible section: ${container.id}`);
                     event.preventDefault(); // Prevent submission
                     alert('Please provide an offensiveness rating for all items you marked as "Stereotype".');
                     if (!firstErrorElement) firstErrorElement = container;
                     // Optionally add visual error indication here
                }
            });

             // If we found an error and prevented submission, focus the first element with an issue
             if (event.defaultPrevented && firstErrorElement) {
                 firstErrorElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                 // Maybe add a temporary highlight to the element
                 firstErrorElement.style.outline = '2px solid red';
                 setTimeout(() => { firstErrorElement.style.outline = ''; }, 3000);
             }
        });
    }

}); // End of DOMContentLoaded