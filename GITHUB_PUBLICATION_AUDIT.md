# GitHub publication audit

## Publication record

- Repository: `Coucou2016/20260519-LarNO-DrainLite`
- Initial visibility at publication: private
- Current visibility: public (changed at the repository owner's request on 2026-09-14)
- Default branch: `main`
- Initial audited commit: `cb2a913560ca668d65e14930dbf0577ce24c9c7f`
- Commit message: `Publish auditable LarNO-DrainLite research snapshot`
- Publication date: 2026-09-14
- Tracked files in the initial commit: 1,324
- Working-file size represented by the commit: approximately 5.80 GiB
- Git LFS files: 139 pointers referring to 130 unique uploaded objects
- Git LFS payload reported by the successful upload: approximately 5.4 GB

## Pre-publication controls

1. Scanned filenames for environment files, private keys, credentials, tokens,
   and certificate material.
2. Scanned source-like text for API-key, password, access-token, and private-key
   patterns. Matches were documentation placeholders or test fixtures only.
3. Excluded all Chrome/browser profiles because they contain local trust-token
   and authentication databases.
4. Excluded caches, temporary builds, third-party installers, duplicate arrays,
   the supplied reference-paper full text, and the duplicate Itzi workspace.
5. Confirmed that no ordinary Git object exceeded GitHub's 100 MB file limit.
6. Routed scientific arrays, estimators, and neural-network weights through Git
   LFS.
7. Re-executed the independent scientific acceptance test immediately before
   publication: `PASS`, 107 of 107 checks.

## Remote verification

The GitHub API reported the same initial commit SHA as the local repository and
returned a complete, non-truncated recursive tree. A second clone was then made
into an unrelated cache directory with automatic LFS download disabled.

Results of the independent clone check:

- cloned commit SHA: `cb2a913560ca668d65e14930dbf0577ce24c9c7f`;
- tracked file count: 1,324;
- clone worktree status: clean;
- test object before `git lfs pull`: 131-byte LFS pointer;
- test object after `git lfs pull`: 224,128-byte NumPy array;
- test object shape: `400 x 560`;
- test object active-cell count: 105,527.

This verifies both the Git repository and the Git LFS retrieval path. It does
not require downloading every 5.4 GB object merely to prove that the remote
commit exists; the successful pre-push LFS batch confirmed all 130 unique
objects, while the independent pull confirms that a new clone can retrieve an
object from that batch.

## Access and licensing note

The repository is publicly readable and can be cloned without authentication.
It contains MIKE-derived reference arrays, an upstream checkpoint, and
road-derived geometry. Public visibility does not replace or expand the reuse
rights of those source materials. Their provenance and known redistribution
boundaries remain documented in `DATA_MANIFEST.md` and
`THIRD_PARTY_NOTICES.md`; downstream users remain responsible for complying
with the applicable source licences and permissions.
