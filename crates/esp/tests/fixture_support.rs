// Copyright 2026 Mark Alan Boykin
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#[test]
fn shared_qualification_snapshots_match_the_supplier_assets() {
    assert_eq!(
        esp_test_support::SMOLLM2_INSTRUCT_V1,
        include_bytes!("../templates/smollm2-instruct-v1.json")
    );
    assert_eq!(
        esp_test_support::QWEN2_MOE_CONFIG_JSON,
        include_str!("support/architectures/qwen2_moe/config.json")
    );
    assert_eq!(
        esp_test_support::QWEN2_MOE_WEIGHTS,
        include_bytes!("support/architectures/qwen2_moe/model.safetensors")
    );
}
