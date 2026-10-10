// Run with Node 22+, a local Chrome debugging endpoint, and an isolated app/database.
// Requires QZSQL_E2E_BASE_URL and QZSQL_E2E_ARTIFACT_DIR. Never targets public hosts.
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const base = new URL(process.env.QZSQL_E2E_BASE_URL);
assert.ok(['localhost', '127.0.0.1'].includes(base.hostname), 'Use an isolated local app.');
const artifacts = process.env.QZSQL_E2E_ARTIFACT_DIR;
assert.ok(artifacts, 'Set QZSQL_E2E_ARTIFACT_DIR to a temporary artifact directory.');
await mkdir(artifacts, { recursive: true });
const targets = await (await fetch('http://127.0.0.1:9222/json')).json();
const ws = new WebSocket(targets.find(target => target.type === 'page').webSocketDebuggerUrl);
await new Promise(resolve => ws.addEventListener('open', resolve, { once: true }));
let sequence = 0;
const pending = new Map();
const errors = [];
const cases = [];
ws.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id) {
        const promise = pending.get(message.id);
        pending.delete(message.id);
        message.error ? promise.reject(message.error) : promise.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
});
function call(method, params = {}) {
    return new Promise((resolve, reject) => {
        const id = ++sequence;
        pending.set(id, { resolve, reject });
        ws.send(JSON.stringify({ id, method, params }));
    });
}
async function evaluate(expression) {
    const result = await call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
}
const click = selector => evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
async function waitFor(expression) {
    for (let attempt = 0; attempt < 60; attempt++) {
        if (await evaluate(expression)) return;
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw Error(`Timed out waiting for ${expression}`);
}
async function navigate(path, ready) {
    await call('Page.navigate', { url: new URL(path, base).href });
    await waitFor(`document.readyState === 'complete' && (${ready})`);
}
async function capture(name) {
    const screenshot = await call('Page.captureScreenshot');
    await writeFile(join(artifacts, name), Buffer.from(screenshot.data, 'base64'));
}
async function saveCases() {
    await writeFile(join(artifacts, 'cases.json'), JSON.stringify(cases, null, 2));
}
async function register(state, marker) {
    await navigate('/', `!!document.getElementById('name')`);
    await evaluate(`document.getElementById('name').value = ${JSON.stringify(marker)};
        document.getElementById('user_state').value = ${JSON.stringify(state)};
        document.getElementById('age').value = '25';
        document.getElementById('female').checked = true;`);
    await click('button[type="submit"]');
    await waitFor(`location.pathname === '/quiz' && !!document.getElementById('begin-survey')`);
    assert.equal(await evaluate(`document.getElementById('quiz-form').hidden`), true);
    await click('#begin-survey');
    assert.equal(await evaluate(`document.querySelectorAll('.quiz-item:not([hidden])').length`), 1);
    assert.equal(await evaluate(`document.getElementById('survey-intro').hidden`), true);
}
async function finishSubmission(marker, expectedCount) {
    const payload = await evaluate(`Object.fromEntries(new FormData(document.getElementById('quiz-form')))`);
    assert.equal(payload.user_name, marker);
    assert.equal(Object.keys(payload).filter(key => key.startsWith('annotation_')).length, expectedCount);
    // Save before submitting so a failed assertion never loses the cleanup marker.
    cases.push({ marker, payload });
    await saveCases();
    await click('#submit-survey');
    await waitFor(`location.pathname === '/thank_you' && document.body.textContent.includes('successfully recorded')`);
}
try {
    await call('Runtime.enable');
    await call('Page.enable');
    await call('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
    const desktopName = 'QZSQL_E2E_' + randomUUID();
    await register('Maharashtra', desktopName);
    assert.equal(await evaluate(`document.querySelectorAll('.group-explanation').length`), 22);
    await capture('desktop-question.png');
    await evaluate(`document.getElementById('survey-help').focus()`);
    await click('#survey-help');
    assert.equal(await evaluate(`document.getElementById('answer-help').open`), true);
    for (const type of ['keyDown', 'keyUp']) {
        await call('Input.dispatchKeyEvent', { type, key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27, nativeVirtualKeyCode: 27 });
    }
    await waitFor(`!document.getElementById('answer-help').open`);
    assert.equal(await evaluate(`document.activeElement.id`), 'survey-help');
    await click('#next-question');
    assert.equal(await evaluate(`document.getElementById('error-0').hidden`), false);
    await click('#anno_s_0');
    await click('#next-question');
    assert.match(await evaluate(`document.getElementById('error-0').textContent`), /rating/);
    await click('#off_4_0');
    await click('#next-question');
    await click('#previous-question');
    assert.equal(await evaluate(`document.getElementById('off_4_0').checked`), true);
    await click('#anno_ns_0');
    assert.equal(await evaluate(`document.getElementById('off_4_0').checked`), false);
    assert.equal(await evaluate(`document.getElementById('off_4_0').disabled`), true);
    await click('#anno_s_0');
    await click('#off_0_0');
    await click('#next-question');
    for (let index = 1; index < 22; index++) {
        const option = ['s', 'ns', 'unsure'][index % 3];
        await click(`#anno_${option}_${index}`);
        if (option === 's') await click(`#off_${Math.floor(index / 3) % 6}_${index}`);
        await click('#next-question');
    }
    assert.equal(await evaluate(`document.getElementById('review-panel').hidden`), false);
    assert.equal(await evaluate(`document.querySelectorAll('#review-list li').length`), 22);
    await click('#review-list li:nth-child(2) button');
    await click('#anno_s_1');
    await click('#next-question');
    assert.equal(await evaluate(`document.getElementById('error-1').hidden`), false);
    await click('#off_5_1');
    await click('#next-question');
    await click('#review-list li:nth-child(3) button');
    await click('#anno_s_2');
    await click('#previous-question');
    await click('#submit-survey');
    assert.equal(await evaluate(`document.getElementById('item-2').hidden`), false);
    await click('#off_3_2');
    await click('#next-question');
    await finishSubmission(desktopName, 22);

    await call('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    const mobileName = 'QZSQL_E2E_' + randomUUID();
    await register('Goa', mobileName);
    assert.equal(await evaluate(`document.documentElement.scrollWidth <= innerWidth`), true);
    assert.equal(await evaluate(`document.querySelector('.answer-card').getBoundingClientRect().height >= 44`), true);
    await capture('mobile-question.png');
    for (let index = 0; index < 18; index++) {
        const option = ['s', 'ns', 'unsure'][index % 3];
        await click(`#anno_${option}_${index}`);
        if (option === 's') await click(`#off_${Math.floor(index / 3) % 6}_${index}`);
        await click('#next-question');
    }
    await finishSubmission(mobileName, 18);

    // The original native form remains fully usable without the guided JavaScript.
    await call('Emulation.setDeviceMetricsOverride', { width: 320, height: 740, deviceScaleFactor: 1, mobile: true });
    await call('Emulation.setScriptExecutionDisabled', { value: true });
    const fallbackName = 'QZSQL_E2E_' + randomUUID();
    await navigate('/quiz?' + new URLSearchParams({ name: fallbackName, state: 'Maharashtra', age: '25', sex: 'Female' }), `!!document.getElementById('quiz-form')`);
    assert.equal(await evaluate(`document.getElementById('quiz-form').hidden`), false);
    assert.equal(await evaluate(`document.querySelectorAll('.quiz-item:not([hidden])').length`), 22);
    assert.equal(await evaluate(`document.documentElement.scrollWidth <= innerWidth`), true);
    for (let index = 0; index < 22; index++) await click(`#anno_unsure_${index}`);
    await finishSubmission(fallbackName, 22);
    await call('Emulation.setScriptExecutionDisabled', { value: false });

    const raw = await fetch(new URL('/admin/download_raw', base));
    assert.equal(raw.status, 200);
    assert.match(raw.headers.get('content-disposition'), /raw_quiz_results.csv/);
    const rawCsv = await raw.text();
    assert.equal(rawCsv.split('\n')[0].trim(), 'id,timestamp,user_name,user_state,user_age,user_sex,category,attribute_superset,annotation,offensiveness_rating');
    for (const testCase of cases) assert.ok(rawCsv.includes(testCase.marker));
    await writeFile(join(artifacts, 'raw.csv'), rawCsv);
    const processed = await fetch(new URL('/admin/download_processed', base));
    assert.equal(processed.status, 200);
    const processedCsv = await processed.text();
    assert.equal(processedCsv.split('\n')[0].trim(), 'Stereotype_State,Category,Attribute,Stereotype_Votes,Not_Stereotype_Votes,Not_Sure_Votes,Average_Offensiveness');
    await writeFile(join(artifacts, 'processed.csv'), processedCsv);
    assert.deepEqual(errors, []);
    console.log('PASS: browser registration → voting → review/edit → real POST → thank-you, desktop/mobile, all annotation and rating values, required-answer validation, help keyboard controls, no-JavaScript submission, both CSV downloads, no runtime errors.');
    console.log('Created 62 synthetic response rows; verify and delete using the recorded markers in cases.json.');
} finally {
    await call('Emulation.setScriptExecutionDisabled', { value: false });
    ws.close();
}
