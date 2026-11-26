"""Editable domain hierarchy, independent of staged decisions."""
import json
from pathlib import Path
import re
import unicodedata

def label(value):
    if (not isinstance(value,str) or not value.strip() or value.strip() in {".",".."}
            or re.search(r"[/\\\x00-\x1f]",value)):
        raise ValueError(f"잘못된 분야 이름: {value!r}")
    return unicodedata.normalize("NFC",value.strip())


class Taxonomy:
    """Editable hierarchy with strict labels and explicit unlisted proposals."""

    def __init__(self,path):
        self.path=Path(path)
        self.data=json.loads(self.path.read_text(encoding="utf-8-sig"))
        if self.data.get("schema_version")!=1 or not isinstance(self.data.get("domains"),dict) or not self.data["domains"]:
            raise ValueError("domains 객체와 schema_version=1이 필요합니다")
        self.domains=self.data["domains"]
        self.aliases={}
        for major,branch in self.domains.items():
            if label(major)!=major or not isinstance(branch,dict) or not isinstance(branch.get("scope"),str) or not branch["scope"].strip():
                raise ValueError(f"대분류 이름·scope를 확인하세요: {major}")
            children=branch.get("subdomains")
            if not isinstance(children,dict) or not children:
                raise ValueError(f"subdomains가 필요합니다: {major}")
            fallback=branch.get("fallback_subdomain")
            if fallback is not None and fallback not in children:
                raise ValueError(f"fallback_subdomain은 등록된 소분류여야 합니다: {major}")
            for minor,definition in children.items():
                if label(minor)!=minor or not isinstance(definition,dict) or not isinstance(definition.get("scope"),str) or not definition["scope"].strip():
                    raise ValueError(f"소분류 이름·scope를 확인하세요: {major}/{minor}")
                criteria=definition.get("core_criteria")
                if not isinstance(criteria,dict) or not criteria or any(
                        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*",k) or not isinstance(v,str) or not v.strip()
                        for k,v in criteria.items()):
                    raise ValueError(f"core_criteria의 영문 ID와 설명을 확인하세요: {major}/{minor}")
                aliases=definition.get("aliases",[])
                if not isinstance(aliases,list) or any(not isinstance(v,str) for v in aliases):
                    raise ValueError("aliases는 문자열 배열이어야 합니다")
                for alias in [minor,*aliases]:
                    alias=label(alias)
                    self.aliases.setdefault(alias,set()).add(major+"/"+minor)
        if not isinstance(self.data.get("policy",{}),dict):
            raise ValueError("policy는 객체여야 합니다")
        if type(self.data.get("policy",{}).get("allow_external_domains",False)) is not bool:
            raise ValueError("allow_external_domains는 true/false여야 합니다")


    def target(self,value):
        path=self.resolve(value)
        major,_,minor=path.partition("/")
        return {"path":path,"major":major,"minor":minor or None}


    def resolve(self, value, external=False):
        if not isinstance(value, str):
            raise ValueError("분야 경로는 문자열이어야 합니다")
        parts = [label(p) for p in value.split("/")]
        if len(parts) > 2:
            raise ValueError("대분류 또는 대분류/소분류가 필요합니다")
        if len(parts) == 1 and parts[0] not in self.domains:
            matches = self.aliases.get(parts[0], set())
            if len(matches) == 1:
                return next(iter(matches))
        major = parts[0]
        if major not in self.domains:
            raise ValueError(f"등록되지 않은 대분류: {major}; 분류체계에 등록하거나 보류하세요")
        if len(parts) == 1:
            return major
        minor = parts[1]
        if minor in self.domains[major]["subdomains"]:
            return major + "/" + minor
        matches = [p for p in self.aliases.get(minor, ()) if p.startswith(major + "/")]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"등록되지 않은 소분류: {value}; proposals에 제안하고 대분류를 보존하세요")


    def context(self, target=None):
        # Membership classification is independent of the requested glossary.
        return {"taxonomy_version": self.data.get("taxonomy_version", "unspecified"),
                "domains": self.domains, "policy": self.data.get("policy", {})}


    def phase_context(self, phase, majors=()):
        if phase == "major":
            return {"majors": {name: {"scope": branch["scope"], "criteria": {"scope": branch["scope"]}}
                               for name, branch in self.domains.items()}}
        return {"confirmed_majors": list(majors),
                "domains": {name: self.domains[name] for name in majors}}


    def criterion(self, path):
        resolved = self.resolve(path)
        major, _, minor = resolved.partition("/")
        return (self.domains[major]["subdomains"][minor]["core_criteria"] if minor
                else {"scope": self.domains[major]["scope"]})


    def proposal(self, path):
        if not isinstance(path, str):
            raise ValueError("제안 분야는 문자열이어야 합니다")
        parts = [label(p) for p in path.split("/")]
        if len(parts) != 2 or parts[0] not in self.domains:
            raise ValueError("새 소분류 제안에는 등록된 대분류/새 소분류가 필요합니다")
        try:
            self.resolve(path)
        except ValueError:
            return "/".join(parts)
        raise ValueError("등록된 분야는 제안이 아니라 assignments에 넣으세요")

