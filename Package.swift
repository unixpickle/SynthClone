// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "SynthClone",
  platforms: [
    .macOS(.v13)
  ],
  products: [],
  dependencies: [
    .package(url: "https://github.com/unixpickle/honeycrisp", from: "0.0.21"),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    .package(url: "https://github.com/vapor/vapor.git", from: "4.102.1"),
  ],
  targets: [
    .executableTarget(
      name: "SynthClone",
      dependencies: [
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
        .product(name: "Vapor", package: "vapor"),
      ],
      resources: [
        .process("Resources")
      ]
    )
  ]
)
