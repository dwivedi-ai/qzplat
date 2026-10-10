document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('quiz-form');
    const items = Array.from(document.querySelectorAll('.quiz-item'));
    if (!form || !items.length) return;

    const practice = document.getElementById('practice');
    const begin = document.getElementById('begin-survey');
    const navigation = document.getElementById('survey-navigation');
    const previous = document.getElementById('previous-question');
    const next = document.getElementById('next-question');
    const submit = document.getElementById('submit-survey');
    const progress = document.getElementById('survey-progress');
    const progressLabel = document.getElementById('progress-label');
    const progressBar = document.getElementById('progress-bar');
    const review = document.getElementById('review-panel');
    const reviewList = document.getElementById('review-list');
    const intro = document.getElementById('survey-intro');
    const help = document.getElementById('answer-help');
    const helpContent = document.getElementById('help-content');
    document.querySelectorAll('#survey-guide > p').forEach(paragraph => {
        helpContent.append(paragraph.cloneNode(true));
    });
    document.getElementById('survey-help').addEventListener('click', () => help.showModal());
    document.getElementById('close-help').addEventListener('click', () => help.close());
    const labels = {
        'Stereotype': 'Yes, I recognise this stereotype',
        'Not a Stereotype': 'No, I don’t recognise this stereotype',
        'Not sure': 'Not sure'
    };
    // Keep Submit beside navigation in the guided flow; its original position
    // remains available when JavaScript is disabled.
    navigation.append(submit);
    let current = 0;
    let editing = false;

    const selectedAnswer = item => item.querySelector('.annotation-options input:checked');
    const selectedRating = item => item.querySelector('.rating-options input:checked');
    const errorFor = item => item.querySelector('.question-error');

    function syncRating(item) {
        const isStereotype = selectedAnswer(item)?.value === 'Stereotype';
        const container = item.querySelector('.offensiveness-rating-container');
        container.hidden = !isStereotype;
        container.querySelectorAll('input').forEach(input => {
            input.required = isStereotype;
            input.disabled = !isStereotype;
            if (!isStereotype) input.checked = false;
        });
    }

    function validate(item, focus = true) {
        const answer = selectedAnswer(item);
        const missingRating = answer?.value === 'Stereotype' && !selectedRating(item);
        const error = errorFor(item);
        error.textContent = !answer ? 'Choose Yes, No, or Not sure to continue.' :
            missingRating ? 'Choose an offensiveness rating from 0 to 5 to continue.' : '';
        error.hidden = !error.textContent;
        if (error.textContent && focus) {
            const target = !answer ? item.querySelector('.annotation-options input') :
                item.querySelector('.rating-options input');
            target.scrollIntoView({ block: 'center' });
            target.focus({ preventScroll: true });
        }
        return !error.textContent;
    }

    function updateProgress() {
        const completed = items.filter(item => selectedAnswer(item) &&
            (selectedAnswer(item).value !== 'Stereotype' || selectedRating(item))).length;
        progressLabel.textContent = current === items.length ?
            `Review answers · ${completed}/${items.length}` :
            `${current + 1} of ${items.length}`;
        progressBar.value = completed;
    }

    function buildReview() {
        reviewList.replaceChildren();
        items.forEach((item, index) => {
            const entry = document.createElement('li');
            const heading = document.createElement('strong');
            heading.textContent = item.querySelector('h2').textContent;
            const answer = document.createElement('p');
            answer.textContent = labels[selectedAnswer(item)?.value] || 'Unanswered';
            if (selectedAnswer(item)?.value === 'Stereotype') {
                answer.textContent += ` · Offensiveness: ${selectedRating(item)?.value ?? 'Unanswered'}/5`;
            }
            const edit = document.createElement('button');
            edit.type = 'button';
            edit.className = 'secondary-button';
            edit.textContent = 'Edit';
            edit.setAttribute('aria-label', `Edit answer for ${heading.textContent}`);
            edit.addEventListener('click', () => {
                editing = true;
                show(index);
            });
            entry.append(heading, answer, edit);
            reviewList.append(entry);
        });
    }

    function show(index, focus = true) {
        current = index;
        const reviewing = index === items.length;
        items.forEach((item, itemIndex) => item.hidden = itemIndex !== index);
        review.hidden = !reviewing;
        submit.hidden = !reviewing;
        previous.textContent = editing ? 'Back to review' : 'Back';
        previous.disabled = index === 0 && !editing;
        next.hidden = reviewing;
        next.textContent = editing ? 'Return to review' :
            index === items.length - 1 ? 'Review answers' : 'Next';
        if (reviewing) buildReview();
        updateProgress();
        if (focus) {
            const heading = reviewing ? document.getElementById('review-title') : items[index].querySelector('h2');
            heading.focus({ preventScroll: true });
            form.scrollIntoView({ block: 'start' });
        }
    }

    items.forEach(item => {
        item.querySelector('.rating-fallback').hidden = true;
        syncRating(item);
        item.addEventListener('change', event => {
            if (event.target.closest('.annotation-options')) {
                syncRating(item);
                if (event.target.value === 'Stereotype') {
                    item.querySelector('.offensiveness-rating-container').scrollIntoView({ block: 'nearest' });
                }
            }
            if (!errorFor(item).hidden) validate(item, false);
            updateProgress();
        });
        item.hidden = true;
    });

    // Custom validation makes hidden questions reachable and focuses their missing answer.
    form.noValidate = true;
    form.hidden = true;
    begin.hidden = false;
    begin.addEventListener('click', () => {
        practice.hidden = true;
        intro.hidden = true;
        form.hidden = false;
        navigation.hidden = false;
        progress.hidden = false;
        show(0);
    });
    next.addEventListener('click', () => {
        if (!validate(items[current])) return;
        const destination = editing ? items.length : current + 1;
        editing = false;
        show(destination);
    });
    previous.addEventListener('click', () => {
        const destination = editing ? items.length : current - 1;
        editing = false;
        show(Math.max(0, destination));
    });
    form.addEventListener('submit', event => {
        // Enter inside the survey must not bypass the remaining questions or review.
        if (current !== items.length) {
            event.preventDefault();
            next.click();
            return;
        }
        const invalid = items.findIndex(item => !validate(item, false));
        if (invalid !== -1) {
            event.preventDefault();
            editing = true;
            show(invalid, false);
            validate(items[invalid]);
        }
    });
    // Reconcile restored radio values when returning through browser history.
    window.addEventListener('pageshow', () => {
        items.forEach(syncRating);
        if (!form.hidden) show(current, false);
    });
});
